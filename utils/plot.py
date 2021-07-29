import matplotlib.pyplot as plt

def plot_iccv_img_onerow(torch_imgs=[], title=[], text=[], text_color='white', figsize=(16, 4),
                         fontsize=12, xy=(50, 10), save_path=None, show=False):
    assert len(torch_imgs)==len(title)
    if torch_imgs[0].shape[1]==1:
        imgs = [img[0].detach().permute(1, 2, 0).cpu().numpy() for img in torch_imgs]
    else:
        imgs = [img.squeeze().detach().permute(1, 2, 0).cpu().numpy() for img in torch_imgs]

    plt.figure(figsize=figsize)

    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs),i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title[i], fontsize=fontsize)
        plt.text(xy[0],xy[1], text[i], fontsize=fontsize, color=text_color)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
