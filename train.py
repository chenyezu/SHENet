import random
import sys

from data import (
    ImageDetectionsField, TextField, GridField, RawField
)
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import (
    Transformer, TransformerEncoder, TransformerDecoder, ScaledDotProductAttention,
    Projector
)
from transformers.optimization import (
    get_constant_schedule_with_warmup, AdamW
)
import torch
from torch import nn
from torch.nn import NLLLoss
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
from tqdm import tqdm
import numpy as np
import math


random.seed(118)
torch.manual_seed(118)
np.random.seed(118)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()

    running_loss = 0.0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        with torch.no_grad():
            for it, data in enumerate(dataloader):
                grid = data["grid"].to(device, non_blocking=True)
                obj = data["object"].to(device, non_blocking=True)
                captions = data["text"].to(device)

                out, _ = model(obj=obj, grid=grid, seq=captions, mode="xe")
                out = out[:, :-1].contiguous()
                captions_gt = captions[:, 1:].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    loss = running_loss / len(dataloader)
    ret = {"loss": loss}

    return ret


def evaluate_metrics(model, dataloader, text_field):
    model.eval()

    gen, gts = {}, {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        with torch.no_grad():
            for it, data in enumerate(dataloader):
                grid = data["grid"].to(device, non_blocking=True)
                obj = data["object"].to(device, non_blocking=True)

                out, _ = model(
                    obj=obj, grid=grid, max_len=20, mode="rl",
                    eos_idx=text_field.vocab.stoi['<eos>'], beam_size=5, out_size=1,
                )

                caps_gen = text_field.decode(out, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(data["text"], caps_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    # scheduler.step()
    running_loss = 0.0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        for it, data in enumerate(dataloader):
            grid = data["grid"].to(device, non_blocking=True)
            obj = data["object"].to(device, non_blocking=True)
            captions = data["text"].to(device, non_blocking=True)

            out, _ = model(obj=obj, grid=grid, seq=captions, mode="xe")
            out = out[:, :-1].contiguous()
            captions_gt = captions[:, 1:].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    scheduler.step()
    loss = running_loss / len(dataloader)
    ret = {"loss": loss}
    
    return ret

def my_decode(word_idxs, eos_idx, join_words=True):
    captions = {}
    for i, wis in enumerate(word_idxs):
        caption = []
        for wi in wis:
            word = int(wi)
            if word == eos_idx:
                break
            caption.append(word)
        # pdb.set_trace()
        if join_words:
            caption = ' '.join(caption)
        captions[i] = [caption]
    return captions


def encode_caps_gt(caps_gts,stoi_for_cider):
    encoded_caps_gt = {}
    for i, caps_gt in enumerate(caps_gts):
        refs = []
        for sentence in caps_gt:
            words = sentence.split(' ')
            try:
                indexes = [stoi_for_cider[word] for word in words]
            except KeyError:
                print('raw sentence')
                print(sentence)
                raise
            refs.append(indexes)
        encoded_caps_gt[i] = refs
    return encoded_caps_gt


def train_scst(model, dataloader, optim, cider, text_field,stoi_for_cider):
    # Training with self-critical
    model.train()
    # temp=0
    #tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    running_loss = .0
    seq_len = 20
    beam_size = 5
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        for it, data in enumerate(dataloader):
            # temp = temp + 1
            # if temp > 10:
            #     break
            grid = data["grid"].to(device, non_blocking=True)
            obj = data["object"].to(device, non_blocking=True)

            out, log_prob = model(
                obj=obj, grid=grid, max_len=seq_len, mode="rl",
                eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=beam_size,
            )
            
            # Rewards
            # caps_gen = text_field.decode(out.view(-1, seq_len))
            # caps_gt = list(itertools.chain(*([c, ] * beam_size for c in data["text"])))
            # caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            # reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            replicated_caps_gt = list(itertools.chain(*([c, ] * beam_size for c in data["text"])))
            my_caps_gen = my_decode(out.view(-1, seq_len), text_field.vocab.stoi['<eos>'], join_words=False)
            my_caps_gt = encode_caps_gt(replicated_caps_gt,stoi_for_cider)
            reward = cider.compute_score(my_caps_gt, my_caps_gen)[1].astype(np.float32) # 直接通过索引计算rewards，节省时间

            reward = torch.from_numpy(reward).to(device).view(obj.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_prob, -1) * (reward - reward_baseline)
            loss = loss.mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
    
    # tokenizer_pool.close()
    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    ret = {
        "loss": loss,
        "reward": reward,
        "reward_baseline": reward_baseline
    }

    return ret

def encode_corpus(corpus, stoi,stoi_for_cider):
    encoded_corpus = {}
    fix_stoi = {}
    stoi_copy = {}  # use copy since stoi is NOT A NORMAL DICT!!!
    for k, v in stoi.items():
        stoi_copy[k] = v

    for key, value in tqdm(corpus.items(), desc='encode corpus'):
        words = value[0].split(' ')
        encoded_words = []
        for word in words:
            index = stoi_copy.get(word, 0)
            if index == 0:
                index = fix_stoi.get(word, None)
                if index is None:
                    fix_stoi[word] = len(stoi_copy) + len(fix_stoi)
                    index = fix_stoi[word]
            encoded_words.append(index)
        encoded_corpus[key] = [encoded_words]

    for k, v in stoi_copy.items():
        stoi_for_cider[k] = v
    for k, v in fix_stoi.items():
        stoi_for_cider[k] = v
    return encoded_corpus


def make_corpus(ref_train):
    res = {}
    for i, item in enumerate(ref_train):
        res[i] = [item]
    return res


def build_cider_train(ref_caps_train,stoi_for_cider ):
    corpus = make_corpus(ref_caps_train)
    encoded_corpus = encode_corpus(corpus, text_field.vocab.stoi,stoi_for_cider)
    cider_train = Cider(encoded_corpus)
    return cider_train

if __name__ == '__main__':
    parser.add_argument('--exp_name', type=str, default='SHENet')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--bs_reduct', type=int, default=5)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--lr_xe', type=float, default=1e-4)
    parser.add_argument('--lr_rl', type=float, default=5e-6)
    parser.add_argument('--wd_rl', type=float, default=0.05)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--annotations', type=str, default="./annotations")
    parser.add_argument('--obj_file', type=str, default="./data/features/vinvl.hdf5")
    parser.add_argument('--grid_file', type=str, default="./data/features/CLIP_features.hdf5") 
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    args = parser.parse_args()
    
    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs")/args.exp_name)
    if not (args.resume_last or args.resume_best):
        shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    print(args)
    print('Transformer Training')

    device = torch.device(args.devices[0])
    writer = SummaryWriter(log_dir=args.save_dir/"tensorboard")

    # Create the dataset
    object_field = ImageDetectionsField(
        obj_file=args.obj_file,
        max_detections=50, preload=args.preload
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
        remove_punctuation=True, nopoints=False
    )
    grid_field = GridField(
        ctx_file=args.grid_file, preload=args.preload
    )

    fields = {
        "object": object_field, "text": text_field, "img_id": RawField(),
        "grid": grid_field
    }
    dset = args.annotations
    dataset = COCO(fields, dset, dset)
    train_dataset, val_dataset, test_dataset = dataset.splits

    fields = {
        "object": object_field, "text": RawField(), "img_id": RawField(),
        "grid": grid_field
    }
    dict_dataset_train = train_dataset.image_dictionary(fields)
    dict_dataset_val = val_dataset.image_dictionary(fields)
    dict_dataset_test = test_dataset.image_dictionary(fields)
    
    ref_caps_train = list(train_dataset.text)
    stoi_for_cider = {}

    
    # build vocabulary
    vocab_file = 'vocab/vocab_coco.pkl'
    if not os.path.isfile(vocab_file):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open(vocab_file, 'wb'))
    else:
        text_field.vocab = pickle.load(open(vocab_file, 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(
        3, 0, 
        attention_module=ScaledDotProductAttention,
        use_gse=True,
        use_hse=True,
        cvf_mode='cross_view',
        # use_gse=False,
        # use_hse=False,
        # num_prototypes=5
        num_semantic_groups=5,
        use_hse_grid=False,
        gse_num_layers=3
    )
    decoder = TransformerDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    projector = Projector(
        f_obj=2048, f_grid=2048,
        f_out=512, drop_rate=0.1)

    model = Transformer(
        bos_idx=text_field.vocab.stoi['<bos>'],
        encoder=encoder, decoder=decoder, projector=projector
    ).to(device)
    model = nn.DataParallel(model, device_ids=args.devices)

    # optimizer
    no_decay = [
        n for n, m in model.named_modules()
        if any(isinstance(m, nd) for nd in [nn.LayerNorm, ])
    ]
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
        {'params': [p for n, p in model.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    def lambda_lr(s):
        base_lr = 0.0001
        print("s:", s)
        if s <= 3:
            lr = base_lr * s / 4
        elif s <= 10:
            lr = base_lr
        elif s <= 12:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        # s += 1
        return lr


    def lambda_rl_lr(s):
        base_lr = args.lr_rl
        lr = base_lr
        # if s <= args.rl_at + 15:
        #     lr = base_lr
        # else:
        #     lr = base_lr * 0.1
        return lr

    def lambda_rl_lr2(s):
        base_lr = args.lr_rl * 0.1
        lr = base_lr
        # if s <= args.rl_at + 15:
        #     lr = base_lr
        # else:
        #     lr = base_lr * 0.1
        return lr

    optim = AdamW(grouped_parameters, lr=1, eps=1e-8)
    scheduler = LambdaLR(optim, lambda_lr)
    cider_train = None

    # Initial conditions
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    best_test_cider = .0
    patience = 0
    start_epoch = 0

    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                drop_last=False)
    dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=math.floor(args.batch_size // 5), shuffle=True,
                                       num_workers=args.workers, drop_last=True)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=math.floor(args.batch_size // 5), shuffle=False,
                                     num_workers=1, drop_last=False)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=math.floor(args.batch_size // 5), shuffle=False,
                                      num_workers=1, drop_last=False)


    # resume training
    if args.resume_last or args.resume_best:
        fname = "ckpt_last.pth" if args.resume_last else "ckpt_best.pth"
        fname = args.save_dir/fname

        if os.path.exists(fname):
            data = torch.load(fname,map_location={'cuda:1':str(device)})
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['model'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            patience = 0
            use_rl = data['use_rl']
            if use_rl:
                scheduler = LambdaLR(optim, lambda_rl_lr)
                for i in range(start_epoch):
                    scheduler.step()
            scheduler.load_state_dict(data['scheduler'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))




    print("Training starts")
    if start_epoch==0:
        scheduler.step()
    for e in range(start_epoch, start_epoch + 100):
        current_lr = optim.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('data/learning_rate', current_lr, e)
        print('lr', current_lr)

        # training epoch
        if not use_rl:
            ret = train_xe(model, dataloader_train, optim, text_field)
            for k, v in ret.items():
                writer.add_scalar(f'data/train_{k}', v, e)
        else:
            if cider_train is None:
                cider_train = build_cider_train(ref_caps_train, stoi_for_cider)
            ret = train_scst(model, dict_dataloader_train, optim, cider_train, text_field,stoi_for_cider)
            for k, v in ret.items():
                writer.add_scalar(f'data/train_{k}', v, e)

        # Validation loss
        ret = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        for k, v in ret.items():
            writer.add_scalar(f'data/val_{k}', v, e)
        val_loss = ret["loss"]

        # Validation scores
        val_scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", val_scores)
        val_cider = val_scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', val_scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', val_scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', val_scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', val_scores['ROUGE'], e)
        writer.add_scalar('data/val_spice', val_scores['SPICE'], e)

        # Test scores
        test_scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", test_scores)
        test_cider = test_scores['CIDEr']
        writer.add_scalar('data/test_cider', test_scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', test_scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', test_scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', test_scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', test_scores['ROUGE'], e)
        writer.add_scalar('data/test_spice', test_scores['SPICE'], e)

        if use_rl:
            scheduler.step()

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1


        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True


        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not \
                                any(nd in n for nd in no_decay)], 'weight_decay': args.wd_rl},
                    {'params': [p for n, p in model.named_parameters() if \
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                optim = AdamW(grouped_parameters, lr=1, eps=1e-8)
                scheduler = LambdaLR(optim, lambda_rl_lr)
                for i in range(e):
                    scheduler.step()
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load(args.save_dir/'ckpt_best.pth')
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['model'], strict=False)
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        if e == 27:
            if current_lr == args.lr_rl: # 即 5e-6
                patience = 0
                grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not \
                                any(nd in n for nd in no_decay)], 'weight_decay': args.wd_rl},
                    {'params': [p for n, p in model.named_parameters() if \
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                optim = AdamW(grouped_parameters, lr=1, eps=1e-8)
                scheduler = LambdaLR(optim, lambda_rl_lr2)
                for i in range(e):
                    scheduler.step()
                print("Reduce Learining Rate to 5e-7")
                if not best_test:
                    data = torch.load(args.save_dir/'ckpt_best_test.pth')
                    torch.set_rng_state(data['torch_rng_state'])
                    torch.cuda.set_rng_state(data['cuda_rng_state'])
                    np.random.set_state(data['numpy_rng_state'])
                    random.setstate(data['random_rng_state'])
                    model.load_state_dict(data['model'], strict=False)
                    print('Resuming from epoch %d, validation loss %f, best cider %f and best test cider %f' % (
                        data['epoch'], data['val_loss'], data['best_cider'],data['best_test_cider']))

        # sys.exit()
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            "val_scores": val_scores,
            "test_scores": test_scores,
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }, args.save_dir/'ckpt_last.pth')

        if best_test:
            copyfile(args.save_dir/'ckpt_last.pth', args.save_dir/'ckpt_best_test.pth')

        if best:
            copyfile(args.save_dir/'ckpt_last.pth', args.save_dir/'ckpt_best.pth')

        if exit_train:
            writer.close()
            break






