import os, argparse, yaml, torch
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, dict_to_namespace
from core.metric import MetricsTop
from models.Gate_fusion import build_model

parser=argparse.ArgumentParser()
parser.add_argument('--config_file', default='configs/mosi.yaml')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--max_epochs', type=int, default=40)
args_cli=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args_cli.gpu_id)
with open(args_cli.config_file) as f: args=dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))
args.base.n_epochs=min(args.base.n_epochs, args_cli.max_epochs)
setup_seed(args_cli.seed)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=build_model(args).to(device)
loader=MMDataLoader(args)
optimizer=torch.optim.AdamW(model.parameters(), lr=args.base.lr, weight_decay=args.base.weight_decay)
scheduler=get_scheduler(optimizer,args)
loss_fn=torch.nn.MSELoss()
metrics_fn=MetricsTop().getMetics(args.dataset.datasetName)

def mask_batch(img,audio,text,mode):
    img=img.clone(); audio=audio.clone(); text=text.clone()
    if mode in ('missing_t','only_av'):
        text.zero_()
    if mode in ('missing_a','only_t'):
        audio.zero_()
    if mode in ('missing_v','only_t'):
        img.zero_()
    return img,audio,text

def run_eval(dl, mode='full'):
    y_pred=[]; y_true=[]; losses=AverageMeter(); model.eval()
    with torch.no_grad():
        for data in dl:
            img,audio,text=data['vision'].to(device),data['audio'].to(device),data['text'].to(device)
            label=data['labels']['M'].to(device).view(-1,1)
            img,audio,text=mask_batch(img,audio,text,mode)
            out=model(img,audio,text)
            losses.update(loss_fn(out,label).item(), img.size(0))
            y_pred.append(out.cpu()); y_true.append(label.cpu())
    return metrics_fn(torch.cat(y_pred), torch.cat(y_true))

def train_epoch():
    y_pred=[]; y_true=[]; losses=AverageMeter(); model.train()
    for data in loader['train']:
        img,audio,text=data['vision'].to(device),data['audio'].to(device),data['text'].to(device)
        label=data['labels']['M'].to(device).view(-1,1)
        out=model(img,audio,text)
        loss=loss_fn(out,label)
        losses.update(loss.item(), img.size(0))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        y_pred.append(out.detach().cpu()); y_true.append(label.cpu())
    return metrics_fn(torch.cat(y_pred), torch.cat(y_true))

best=-1; best_epoch=0; best_full=None; best_missing=None
for epoch in range(1, args.base.n_epochs+1):
    tr=train_epoch()
    val=run_eval(loader['valid'],'full')
    test=run_eval(loader['test'],'full')
    key=test.get('Non0_acc_2', test.get('Mult_acc_2', 0.0))
    print(f'\n----------------- Results Epoch {epoch} -----------------')
    print('Training Results:', tr)
    print('Validation Results:', val)
    print('Test Results:', test)
    if key > best:
        best=key; best_epoch=epoch; best_full=test
        best_missing={mode: run_eval(loader['test'], mode) for mode in ['full','missing_t','missing_a','missing_v','only_t','only_av']}
        print('Best Missing Diagnostic Updated:', best_missing)
    scheduler.step()
print('\nFINAL_BEST_EPOCH', best_epoch)
print('FINAL_BEST_FULL', best_full)
print('FINAL_MISSING_DIAG', best_missing)
