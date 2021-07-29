import os
import csv
from datetime import datetime

# --------------------------------
# logger
# --------------------------------
class LOG(object):
    def __init__(self, filepath, filename, field_name):
        self.filepath = filepath
        self.filename = filename
        self.field_name = field_name

        self.logfile, self.logwriter = csv_log(file_name=os.path.join(filepath, filename+'.csv'), field_name=field_name)
        self.logwriter.writeheader()

    def record(self, *args):
        dict = {}
        for i in range(len(self.field_name)):
            dict[self.field_name[i]]=args[i]
        self.logwriter.writerow(dict)

    def close(self):
        self.logfile.close()

    def print(self, msg):
        logT(msg)

def csv_log(file_name, field_name):
    assert file_name is not None
    assert field_name is not None
    logfile = open(file_name, 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=field_name)
    return logfile, logwriter

def logT(*args, **kwargs):
     print(get_timestamp(), *args, **kwargs)

def get_timestamp():
    return datetime.now().strftime('%y-%m-%d-%H:%M:%S')


# --------------------------------
# meters
# --------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'