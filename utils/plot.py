import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from .metric import cal_psnr

def plot_csv(*csv_path, legend=[], line_style=['r-'], xy_items=['iter', 'loss'],font_size=0, line_width=2, title='figure',
             root_dir, alpha, fig_name=None, suffix='.pdf', if_log_loss=False, xylabel=['epoch', '$\log(loss)$'], save=False, show=False):
    if legend is not None:
        if isinstance(csv_path, (tuple, list)):
            csv_path=csv_path[0]
        assert len(xy_items)-1==len(line_style)
    font=None if font_size==0 else {'size':font_size}
    plt.figure()
    i = 0
    for path in csv_path:
        print(i)
        print(path)
        data = pd.read_csv(path)
        data.head()
        data.shape
        iter = pd.to_numeric(data[xy_items[0]].values)
        for j in range(len(xy_items)-1):
            loss = pd.to_numeric(data[xy_items[j+1]].values)
            if if_log_loss:
                plt.plot(iter, np.log(alpha[j]*loss), line_style[j], lw=line_width)
            else:
                plt.plot(iter, alpha[j]*loss, line_style[j], lw=line_width)
        i = i+1
    plt.legend(legend, loc='upper right') if len(legend) >0 else None
    plt.grid()
    plt.xlabel(xylabel[0], fontdict=font)
    plt.ylabel(xylabel[1], fontdict=font)
    plt.title(title, fontdict=font)
    if save:
        assert fig_name is not None
        plt.savefig(os.path.join(root_dir, fig_name+suffix), dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_iccv_img(six_torch_imgs, title=[r'$x^(1)$', 'a', 'a','a','a','a'], nRow=3, nCol=12, figsize=(16, 4), p1=(90,30), p2=(120, 100), w1=30, h1=30, w2=30,h2=30,
                  color0='red', color1='red', color2='yellow', linewidth=1.5, cmap='gray', clim=None, save_path=None, show=False, resolution=128, text=[], ylable=''):
    import matplotlib.gridspec as gridspec

    imgs = [img.squeeze().detach().cpu().numpy() for img in six_torch_imgs]

    plt.figure(figsize=figsize)
    plt.axis('off')
    gs = gridspec.GridSpec(nRow, nCol)


    big = [plt.subplot(gs[0:nRow-1, i*2:(i+1)*2]) for i in range(len(six_torch_imgs))]
    sub_left = [plt.subplot(gs[nRow - 1, i * 2]) for i in range(len(six_torch_imgs))]
    sub_right = [plt.subplot(gs[nRow - 1, i * 2 + 1]) for i in range(len(six_torch_imgs))]

    for i in range(len(six_torch_imgs)):

        if i>0:

            big[i].add_patch(plt.Rectangle(p1, w1, h1, fill=False, edgecolor=color1, linewidth=linewidth))
            big[i].add_patch(plt.Rectangle(p2, w2, h2, fill=False, edgecolor=color2, linewidth=linewidth))

        img=big[i].imshow(imgs[i], cmap=plt.cm.get_cmap(cmap) if cmap is not None else None)
        big[i].set_title(title[i])

        if i==0:
            big[i].set_ylabel(ylable)

        # print('big', len(big), len(text))
        if i >0 and i!=len(six_torch_imgs)-1:#i >0 and i!=1:
            # print(i, six_torch_imgs[0].shape, six_torch_imgs[i].shape)

            # psnr = cal_psnr(six_torch_imgs[-1], six_torch_imgs[i])
            if resolution==128:
                psnr = cal_psnr(six_torch_imgs[-1], six_torch_imgs[i])
                big[i].text(97, 9, '{:.3f}'.format(psnr), fontsize=12, color=color0)#CT
            if resolution==256:
                big[i].text(190, 20, '{}'.format(text[i]), fontsize=12, color=color0)#CT
                # big[i].text(100, 20, '{}'.format(text[i]), fontsize=12, color=color0)  # CT
            if resolution==512:
                big[i].text(97, 9, '{:.3f}'.format(cal_psnr(six_torch_imgs[-1], six_torch_imgs[i])), fontsize=12, color=color0)#CT#big[i].text(400, 20, '{}'.format(text[i]), fontsize=12, color=color0)#512x512
		

        # if i==0:
        #     plt.colorbar(img, ax=big[i], shrink=colorbar_shrink)
        if clim is not None:
            img.set_clim(clim)

        if i>0:
            sub_left[i].add_patch(plt.Rectangle((0,0), 0.97*w1, 0.97*h1, fill=False, edgecolor=color1, linewidth=linewidth))
            sub_right[i].add_patch(plt.Rectangle((0,0), 0.97*w2, 0.97*h2, fill=False, edgecolor=color2, linewidth=linewidth))

            sub_left[i].imshow(imgs[i][p1[1]:p1[1]+h1, p1[0]:p1[0]+w1], cmap=plt.cm.get_cmap(cmap))
            sub_right[i].imshow(imgs[i][p2[1]:p2[1]+h2, p2[0]:p2[0]+w2], cmap=plt.cm.get_cmap(cmap))

        big[i].axis('off')
        sub_left[i].axis('off')
        sub_right[i].axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # plt.subplots_adjust(wspace=0.05, hspace=0)
    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()


def plot_iccv_img_onerow(torch_imgs=[], title=[], text=[], text_color='white', figsize=(16, 4), save_path=None, show=False):
    assert len(torch_imgs)==len(title)
    imgs = [img.squeeze().detach().permute(1, 2, 0).cpu().numpy() for img in torch_imgs]
    plt.figure(figsize=figsize)

    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs),i+1)
        plt.imshow(img)
        plt.title(title[i], fontsize=12)
        plt.text(460,40, text[i], fontsize=12, color=text_color)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path)
    if show:
        plt.show()
