import matplotlib.pyplot as plt
import os

def plot_data(x_train, y_train, idx_train, img_x_train, save_dir='/home/dfredin/gwgasf/results/figures/'):
    os.makedirs(save_dir, exist_ok=True)
    
    def save_plot(data, label, title, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
    
    def save_image(data, label, title, filename):
        plt.figure(figsize=(10, 6))
        plt.imshow(data, cmap='rainbow', origin='lower')
        plt.title(title)
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    # Glitch Plot
    y_train_label = str(y_train[0])
    save_plot(x_train[idx_train, :, 0][0], y_train_label, f'Label: {y_train_label}', f'plot_{y_train_label}.png')
    save_image(img_x_train[:,0,:,:][0], y_train_label, f'Label: {y_train_label}', f'gasf_plot_{y_train_label}.png')

    # Signal Plot
    y_train_label = str(y_train[548])
    save_plot(x_train[idx_train,:,0][548], y_train_label, f'Label: {y_train_label}', f'plot_{y_train_label}.png')
    save_image(img_x_train[:,0,:,:][548], y_train_label, f'Label: {y_train_label}', f'gasf_plot_{y_train_label}.png')

    # Background Plot
    y_train_label = str(y_train[55])
    save_plot(x_train[idx_train,:,0][55], y_train_label, f'Label: {y_train_label}', f'plot_{y_train_label}.png')
    save_image(img_x_train[:,0,:,:][55], y_train_label, f'Label: {y_train_label}', f'gasf_plot_{y_train_label}.png')
