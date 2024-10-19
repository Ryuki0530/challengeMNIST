import torch
import torch.nn as nn
import torch.nn.functional as F  # これを追加
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# CNNを使用したニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # F.relu を使用
        x = F.relu(self.conv2(x))  # F.relu を使用
        x = F.max_pool2d(x, 2)     # F.max_pool2d を使用
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))    # F.relu を使用
        x = self.fc2(x)
        return x

# 保存したモデルを読み込む
net = Net()
net.load_state_dict(torch.load('mnist_model.pth'))
net.eval()

# 画像を前処理して推論する関数
def predict_image(image_path):
    img = Image.open(image_path).convert('L')  # グレースケールに変換
    img = img.resize((28, 28))  # サイズを28x28にリサイズ
    img = img.point(lambda x: 0 if x < 128 else 255, '1')  # 閾値処理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    img_tensor = transform(img).unsqueeze(0)

    # 推論
    with torch.no_grad():
        output = net(img_tensor)
        _, predicted = torch.max(output, 1)
    messagebox.showinfo("推論結果", f"予測された数字: {predicted.item()}")

# ファイル選択ダイアログを表示して画像を選択
def open_file_dialog():
    file_path = filedialog.askopenfilename(title="画像ファイルを選択してください", filetypes=[("PNGファイル", "*.png"), ("すべてのファイル", "*.*")])
    if file_path:
        display_image(file_path)  # 画像を表示
    return file_path

# リサイズされた画像をウィンドウで表示
def display_image(image_path):
    global selected_image_path
    selected_image_path = image_path
    img = Image.open(image_path).convert('L')
    img = img.resize((280, 280))
    img_tk = ImageTk.PhotoImage(img)

    label_image.config(image=img_tk)
    label_image.image = img_tk
    label_image.pack()

    btn_run.config(state=tk.NORMAL)
    btn_run.pack()

# 実行ボタンが押されたら推論を開始
def on_run_button():
    global selected_image_path
    if selected_image_path:
        predict_image(selected_image_path)

# メイン処理
if __name__ == '__main__':
    root = tk.Tk()
    root.title("PyTorch MNIST Handwriting Recognition")

    label_image = tk.Label(root)
    
    btn_select = tk.Button(root, text="画像を選択", command=lambda: open_file_dialog())
    btn_select.pack()

    btn_run = tk.Button(root, text="推論を実行", state=tk.DISABLED, command=on_run_button)
    
    selected_image_path = None
    root.mainloop()
