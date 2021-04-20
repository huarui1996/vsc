import argparse
import torch
import yaml
import os
import tools
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.VSC import VSCModel
from models.CaptionGen import CaptionGenerationModule
import utils
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from dataloader import VideoDataset, SummaryDataset


def combine_loss(losses, weights):
    loss = None
    for i in range(len(losses)):
        if loss is None:
            loss = losses[i] * weights[i]
        else:
            loss = loss + losses[i] * weights[i]
    return loss


def pretrain(loader, model, crit, optimizer, lr_scheduler, opt):
    model.train()
    writer = SummaryWriter(log_dir=os.path.join(opt["checkpoint_path"]))
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()
        iteration = 0
        for data in loader:
            fc_feats = data['fc_feats'].to(opt['device'])
            labels = data['labels'].to(opt['device'])
            masks = data['masks'].to(opt['device'])

            optimizer.zero_grad()
            seq_probs, _ = model(fc_feats, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            iteration += 1

            print("iter %d (epoch %d), train_loss = %.6f" %
                    (iteration, epoch, train_loss))

        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (epoch))
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            writer.add_scalar("Pretrain Loss: ", train_loss, epoch)


def test(dataset, model):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    vocab = dataset.get_vocab()
    results = {'usr': [], 'msr': [], 'cap': [], 'target': []}
    for data in loader:
        feats = data['features'].to(args['device'])
        label = data['labels'].to(args['device'])
        usr = data['user_summary']
        cps = data['change_points']
        fps = data['fps']
        picks = data['picks']
        true_len = int(data['true_len'])
        true_seg = int(data['true_seg'])
        true_user = int(data['true_user'])
        true_frame = int(data['true_frame'])
        return_dict = model(feats, label, 'inference')
        cps = cps.numpy()[:, :true_seg].squeeze()
        fps = fps.numpy()[:, :true_seg].squeeze()
        score = return_dict['att'].cpu().numpy().squeeze()[:true_len]
        picks = picks.numpy()[:, :true_len].squeeze()
        usr = usr.numpy().squeeze()[:true_user, :true_frame]
        summary = tools.generate_summary(score,
                                         cps, true_len,
                                         fps, picks)
        results['msr'].append(summary)
        results['usr'].append(usr)
        results['cap'].append(utils.decode_sequence(vocab, return_dict['out_cap']))
        #print(utils.decode_sequence(vocab, label[1:]))
        results['target'].append(utils.decode_sequence(vocab, label[:, 1:]))
        sum_f_score = 0.0
        num = 0.0
    for i in range(len(results['usr'])):
        final_f_score, final_prec, final_rec = tools.evaluate_summary(machine_summary=results['msr'][i],
                                                                      user_summary=results['usr'][i])
        num = num + 1
        sum_f_score = sum_f_score + final_f_score
        print("F score: %.4f, Precision: %.4f, Recall: %.4f" % (final_f_score, final_prec, final_rec))

    ave_f_score = sum_f_score/num
    print("acverage F score: %.4f" % ave_f_score)

    with open("pred.txt", 'w') as f:
        for i in results["cap"]:
            if i:
                f.writelines(i[0] + '\n')
    with open("target.txt", 'w') as f:
        for i in results["target"]:
            if i:
                f.writelines(i[0] + '\n')


def test_pretrain(dataset, model, args):
    loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
    vocab = dataset.get_vocab()
    sens = []
    target = []
    for data in loader:
        # forward the model to get loss
        fc_feats = data['fc_feats'].to(args['device'])
        labels = data['labels'].to(args['device'])
        masks = data['masks'].to(args['device'])
        # video_ids = data['video_ids']

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(fc_feats, mode='inference')

        sents = utils.decode_sequence(vocab, seq_preds)
        true_sen = utils.decode_sequence(vocab, labels[:, 1:])
        sens += sents
        target += true_sen
    f = open("pretrain_caption.txt", 'w')
    t = open("pretrain_target.txt", 'w')
    for i, j in zip(sens, target):
        f.writelines(i + '\n')
        t.writelines(j + '\n')
    f.close()
    t.close()


def train(loader, args, vocab, model, optimizer, lr_scheduler, config):
    writer = SummaryWriter(log_dir=os.path.join(args["checkpoint_path"]))
    if not config['ve']['kl_loss']:
        loss_vis = torch.nn.MSELoss().to(args['device'])
    else:
        loss_vis = nn.KLDivLoss().to(args['device'])
    loss_gen = utils.LanguageModelCriterion().to(args['device'])
    loss_sum = nn.MSELoss().to(args['device'])

    max_epoch = int(args['epochs'])
    att, return_dict, label, example = None, None, None, None
    for epoch in range(max_epoch):
        lr_scheduler.step()
        loss_gen_value, loss_vis_value, loss_sum_value = 0.0, 0.0, 0.0
        iteration = 0
        for data in loader:
            feats = data['features'].to(args['device'])
            label = data['labels'].to(args['device'])
            gt_score = data['gt_score'].to(args['device'])
            mask = data['masks'].to(args['device'])
            optimizer.zero_grad()
            return_dict = model(feats, label)
            if model.VisualEmb.kl:
                vec_v, vec_t = return_dict['p_v'], return_dict['p_t']
            else:
                vec_v, vec_t = return_dict['f_v_'], return_dict['f_t_']
            att = return_dict['att']
            lg = loss_gen(return_dict['prob'], label[:, 1:], mask[:, 1:])
            lv = loss_vis(vec_v, vec_t)
            ls = loss_sum(att, gt_score)
            loss = combine_loss([lv, lg, ls],
                                [config['loss']['weight_vis'],
                                 config['loss']['weight_gen'],
                                 config['loss']['weight_sum']])
            loss.backward()
            optimizer.step()
            loss_gen_value += config['loss']['weight_gen'] * lg.item()
            loss_vis_value += config['loss']['weight_vis'] * lv.item()
            loss_sum_value += config['loss']['weight_sum'] * ls.item()
            train_loss = config['loss']['weight_gen'] * lg.item() + \
                         config['loss']['weight_vis'] * lv.item() + \
                         config['loss']['weight_sum'] * ls.item()
            iteration += 1
            print("iter %d (epoch %d), train_loss = %.6f" %
                    (iteration, epoch, train_loss))
        torch.save(model.state_dict(), os.path.join(args['checkpoint_path'], 'model.pth'))
        # tensorboard data.
        writer.add_scalar("Visual Semantic Model Loss: ", loss_vis_value, epoch)
        writer.add_scalar("Caption Generation Model Loss: ", loss_gen_value, epoch)
        writer.add_scalar("Video Summary Loss: ", loss_sum_value, epoch)
        writer.add_image("Attention Matrix: ", att.view(1, -1, config['vid_len']), epoch)


def main(args):
    if args['mode'] == 'pretrain':
        with open(args['conf'], 'r') as f:
            params = yaml.load(f)
        dataset = VideoDataset(args, 'train')
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
        emb = torch.nn.Embedding(dataset.get_vocab_size(), params['word_dim'])
        model = CaptionGenerationModule(
            vocab_size=dataset.get_vocab_size(),
            max_len=params['max_sen'],
            dim_word=params['word_dim'],
            dim_vid=params['vid_dim'],
            dim_hidden=params['cap']['hidden'],
            embedding=emb,
            n_layers=params['cap']['num_layer'],
            rnn_cell=params['cap']['rnn_type'],
            rnn_dropout_p=params['cap']['rnn_dropout'],
            device=args['device'])
        model.to(device=args['device'])
        crit = utils.LanguageModelCriterion()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"])
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args["learning_rate_decay_every"],
            gamma=args["learning_rate_decay_rate"])
        pretrain(dataloader, model, crit, optimizer, lr_scheduler, args)

    elif args['mode'] == 'train':
        with open(args['conf'], 'r') as f:
            params = yaml.load(f)
        dataset = SummaryDataset(args, 'train', params)
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
        model = VSCModel(
            model_type='custom',
            vocab_size=dataset.get_vocab_size(),
            max_sen=params['max_sen'],
            max_vid=params['vid_len'],
            word_dim=params['word_dim'],
            vid_dim=params['vid_dim'],
            device=args['device'],
            params=params)
        model.CapGen.load_state_dict(torch.load(args['pretrain_model']))
        model.to(device=args['device'])
        optimizer = optim.Adam(
            model.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"])
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args["learning_rate_decay_every"],
            gamma=args["learning_rate_decay_rate"])
        train(dataloader, args, dataset.get_vocab(), model, optimizer, lr_scheduler, params)

    elif args['mode'] == 'test_pretrain':
        with open(args['conf'], 'r') as f:
            params = yaml.load(f)
        dataset = SummaryDataset(args, 'test', params)
        with torch.no_grad():
            model = CaptionGenerationModule(
                vocab_size=dataset.get_vocab_size(),
                max_len=params['max_sen'],
                dim_word=params['word_dim'],
                dim_vid=params['vid_dim'],
                dim_hidden=params['cap']['hidden'],
                n_layers=params['cap']['num_layer'],
                rnn_cell=params['cap']['rnn_type'],
                rnn_dropout_p=params['cap']['rnn_dropout'],
                device=args['device'])
            model.load_state_dict(torch.load(args['pretrain_model']))
            model.to(device=args['device'])
            test_pretrain(dataset, model, args)

    elif args['mode'] == 'test':
        with open(args['conf'], 'r') as f:
            params = yaml.load(f)
        with torch.no_grad():
            dataset = SummaryDataset(args, 'test', params)
            model = VSCModel(
                model_type='custom',
                vocab_size=dataset.get_vocab_size(),
                max_sen=params['max_sen'],
                max_vid=params['vid_len'],
                word_dim=params['word_dim'],
                vid_dim=params['vid_dim'],
                device=args['device'],
                params=params).to(args['device'])
            model.load_state_dict(torch.load(args['model']))
            test(dataset, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_json',
        type=str,
        default='G:/dataset/msvd/all_info.json',
        help='path to the json file containing msvd video info')
    parser.add_argument(
        '--vid_json',
        type=str,
        default='summary_data/info.json',
        help='path to the json file containing msvd video info')
    parser.add_argument(
        '--info_json',
        type=str,
        default='out_path/info.json',
        help='path to the json file containing additional info and vocab')
    parser.add_argument(
        '--caption_json',
        type=str,
        default='out_path/caption.json',
        help='path to the processed video caption json')

    parser.add_argument(
        '--pretrain_model',
        type=str,
        default='out_path/model_515.pth',
        help='path to pretrain model.')

    parser.add_argument(
        '--feats_dir',
        nargs='*',
        type=str,
        default=['feats/'],
        help='path to the directory containing the preprocessed fc feats')

    parser.add_argument(
        '--conf',
        type=str,
        default='vsc_exp_0.yaml',
        help='path to the yaml config file.'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='pretrain',
        help='one of [pretrain, train, test].'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='out_path/model.pth',
        help='path of joint model.'
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=30,
        help='max length of captions(containing <sos>,<eos>)')

    # Optimization: General

    parser.add_argument(
        '--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=5,  # 5.,
        help='clip gradients at this value')

    parser.add_argument(
        '--learning_rate', type=float, default=4e-4, help='learning rate')

    parser.add_argument(
        '--learning_rate_decay_every',
        type=int,
        default=200,
        help='every how many iterations thereafter to drop LR?(in epoch)')

    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)

    parser.add_argument(
        '--optim_alpha', type=float, default=0.9, help='alpha for adam')

    parser.add_argument(
        '--optim_beta', type=float, default=0.999, help='beta used for adam')

    parser.add_argument(
        '--optim_epsilon',
        type=float,
        default=1e-8,
        help='epsilon that goes into denominator for smoothing')

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4,
        help='weight_decay. strength of weight regularization')

    parser.add_argument(
        '--save_checkpoint_every',
        type=int,
        default=5,
        help='how often to save a model checkpoint (in epoch)?')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='out_path',
        help='directory to store checkpointed models')

    parser.add_argument(
        '--device', type=str, default='cuda:0', help='gpu device number')

    args = parser.parse_args()
    args = vars(args)
    main(args)