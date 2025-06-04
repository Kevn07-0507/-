

import tkinter as tk
from tkinter import filedialog, ttk
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from torchaudio.transforms import Resample
import os

class ProfessionalAudioSeparator:
    def __init__(self, master):
        self.master = master
        master.title("人声分离工具")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 使用兼容性更好的预训练模型
        self.bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
        self.sample_rate = self.bundle.sample_rate  # 44100 Hz

        self.model = self.bundle.get_model().to(self.device).eval()

        # 初始化音频数据
        self.original_waveform = None
        self.vocal_waveform = None

        # 初始化界面
        self.create_widgets()
        self.setup_visuals()

    def create_widgets(self):
        # 控制面板
        control_frame = ttk.Frame(self.master)
        control_frame.pack(fill=tk.X, padx=15, pady=15)

        # 文件选择组件
        self.btn_select = ttk.Button(
            control_frame,
            text="选择音频文件",
            command=self.select_file
        )
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.lbl_file = ttk.Label(control_frame, text="未选择文件")
        self.lbl_file.pack(side=tk.LEFT, padx=5)

        # 处理按钮
        self.btn_process = ttk.Button(
            control_frame,
            text="深度分离",
            command=self.process_audio,
            state=tk.DISABLED
        )
        self.btn_process.pack(side=tk.LEFT, padx=5)

        # 播放控制
        play_frame = ttk.Frame(control_frame)
        play_frame.pack(side=tk.LEFT, padx=20)

        self.btn_play_original = ttk.Button(
            play_frame,
            text="▶ 原音",
            command=lambda: self.play_audio(self.original_waveform),
            state=tk.DISABLED
        )
        self.btn_play_original.pack(side=tk.LEFT, padx=5)

        self.btn_play_vocal = ttk.Button(
            play_frame,
            text="▶ 人声",
            command=lambda: self.play_audio(self.vocal_waveform),
            state=tk.DISABLED
        )
        self.btn_play_vocal.pack(side=tk.LEFT, padx=5)

        # 保存按钮
        self.btn_save = ttk.Button(
            control_frame,
            text="💾 保存结果",
            command=self.save_results,
            state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # 频谱图区域
        fig_frame = ttk.Frame(self.master)
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 统一频谱图尺寸
        self.fig_original = plt.Figure(figsize=(8, 4.5), dpi=120)
        self.ax_original = self.fig_original.add_subplot(111)
        self.canvas_original = FigureCanvasTkAgg(self.fig_original, fig_frame)
        self.canvas_original.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_separated = plt.Figure(figsize=(8, 4.5), dpi=120)
        self.ax_separated = self.fig_separated.add_subplot(111)
        self.canvas_separated = FigureCanvasTkAgg(self.fig_separated, fig_frame)
        self.canvas_separated.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def setup_visuals(self):
        """配置专业可视化参数"""
        plt.rcParams.update({
            'font.family': 'Arial',
            'axes.grid': True,
            'grid.linestyle': ':',
            'image.cmap': 'magma',
            'figure.facecolor': '#F5F5F5'
        })

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("音频文件", "*.mp3 *.wav *.flac")])
        if file_path:
            self.file_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path))
            self.btn_process.config(state=tk.NORMAL)

    def process_audio(self):
        self.btn_process.config(state=tk.DISABLED)
        self.master.config(cursor="watch")
        self.master.update()

        try:
            # 加载并预处理音频
            waveform, sample_rate = self.load_and_preprocess()

            # 使用深度学习模型分离人声
            with torch.no_grad():
                mixture = waveform.unsqueeze(0).to(self.device)
                sources = self.model(mixture)

                if sources.shape[1] >= 4:
                    vocal_idx = 3
                else:
                    vocal_idx = 0

                vocal_waveform = sources[0, vocal_idx].mean(dim=0).cpu().numpy()
                original_mono = waveform.mean(dim=0).cpu().numpy()

            # 存储结果
            self.original_waveform = original_mono
            self.vocal_waveform = vocal_waveform

            # 更新界面
            self.update_visualization()
            self.btn_play_original.config(state=tk.NORMAL)
            self.btn_play_vocal.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)

        except Exception as e:
            print(f"处理出错: {str(e)}")
        finally:
            self.btn_process.config(state=tk.NORMAL)
            self.master.config(cursor="")

    def load_and_preprocess(self):
        """稳健的音频加载与预处理"""
        try:
            waveform, sample_rate = torchaudio.load(self.file_path)

            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            if sample_rate != self.sample_rate:
                resampler = Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            return waveform, self.sample_rate

        except Exception as e:
            raise RuntimeError(f"音频加载失败: {str(e)}")

    def update_visualization(self):
        """稳健的频谱可视化"""
        try:
            # 清空原有内容
            self.ax_original.clear()
            self.ax_separated.clear()

            # 原始音频频谱
            if self.original_waveform is not None and len(self.original_waveform) > 512:
                S_orig = librosa.amplitude_to_db(
                    np.abs(librosa.stft(self.original_waveform, n_fft=2048)),
                    ref=np.max
                )
                librosa.display.specshow(
                    S_orig,
                    sr=self.sample_rate,
                    x_axis='time',
                    y_axis='log',
                    ax=self.ax_original
                )
                self.ax_original.set_title("原始频谱图", fontsize=12, pad=15)
                if self.ax_original.images:
                    self.fig_original.colorbar(self.ax_original.images[0],
                                               ax=self.ax_original,
                                               format='%+2.0f dB')

            # 人声音谱
            if self.vocal_waveform is not None and len(self.vocal_waveform) > 512:
                S_vocal = librosa.amplitude_to_db(
                    np.abs(librosa.stft(self.vocal_waveform, n_fft=2048)),
                    ref=np.max
                )
                librosa.display.specshow(
                    S_vocal,
                    sr=self.sample_rate,
                    x_axis='time',
                    y_axis='log',
                    ax=self.ax_separated
                )
                self.ax_separated.set_title("分离人声频谱", fontsize=12, pad=15)
                if self.ax_separated.images:
                    self.fig_separated.colorbar(self.ax_separated.images[0],
                                                ax=self.ax_separated,
                                                format='%+2.0f dB')

            # 统一颜色范围
            if self.ax_original.images and self.ax_separated.images:
                vmin = min(S_orig.min(), S_vocal.min())
                vmax = max(S_orig.max(), S_vocal.max())
                self.ax_original.images[0].set_clim(vmin, vmax)
                self.ax_separated.images[0].set_clim(vmin, vmax)

            self.canvas_original.draw()
            self.canvas_separated.draw()

        except Exception as e:
            print(f"可视化错误: {str(e)}")
            self.clear_visualization()

    def clear_visualization(self):
        """清空可视化内容"""
        self.ax_original.clear()
        self.ax_separated.clear()
        self.canvas_original.draw()
        self.canvas_separated.draw()

    def play_audio(self, audio_data):
        try:
            if audio_data is not None:
                gain = 0.9 / (np.max(np.abs(audio_data)) + 1e-7)
                sd.play(audio_data * gain, self.sample_rate)
        except Exception as e:
            print(f"播放失败: {str(e)}")

    def save_results(self):
        """高分辨率保存频谱图"""
        try:
            if not hasattr(self, 'original_waveform') or not hasattr(self, 'vocal_waveform'):
                print("没有可保存的数据")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[
                    ("WAV 音频", "*.wav"),
                    ("FLAC 音频", "*.flac"),
                    ("频谱图 PNG", "*.png")
                ]
            )
            if not file_path:
                return

            if file_path.endswith(('.wav', '.flac')):
                torchaudio.save(
                    file_path,
                    torch.from_numpy(self.vocal_waveform).unsqueeze(0),
                    self.sample_rate
                )
            else:
                base, ext = os.path.splitext(file_path)

                # 保存原始频谱图
                if self.original_waveform is not None and len(self.original_waveform) > 512:
                    save_fig_orig = plt.figure(figsize=(16, 9), dpi=300)
                    save_ax_orig = save_fig_orig.add_subplot(111)
                    S_orig = librosa.amplitude_to_db(
                        np.abs(librosa.stft(self.original_waveform, n_fft=2048)),
                        ref=np.max
                    )
                    librosa.display.specshow(S_orig,
                                             sr=self.sample_rate,
                                             x_axis='time',
                                             y_axis='log',
                                             ax=save_ax_orig)
                    save_ax_orig.set_title("Original Spectrogram")
                    if save_ax_orig.images:
                        save_fig_orig.colorbar(save_ax_orig.images[0],
                                               ax=save_ax_orig,
                                               format='%+2.0f dB')
                    save_fig_orig.savefig(f"{base}_original.png", bbox_inches='tight')
                    plt.close(save_fig_orig)

                # 保存人声音谱图
                if self.vocal_waveform is not None and len(self.vocal_waveform) > 512:
                    save_fig_vocal = plt.figure(figsize=(16, 9), dpi=300)
                    save_ax_vocal = save_fig_vocal.add_subplot(111)
                    S_vocal = librosa.amplitude_to_db(
                        np.abs(librosa.stft(self.vocal_waveform, n_fft=2048)),
                        ref=np.max
                    )
                    librosa.display.specshow(S_vocal,
                                             sr=self.sample_rate,
                                             x_axis='time',
                                             y_axis='log',
                                             ax=save_ax_vocal)
                    save_ax_vocal.set_title("Vocal Spectrogram")
                    if save_ax_vocal.images:
                        save_fig_vocal.colorbar(save_ax_vocal.images[0],
                                                ax=save_ax_vocal,
                                                format='%+2.0f dB')
                    save_fig_vocal.savefig(f"{base}_vocal.png", bbox_inches='tight')
                    plt.close(save_fig_vocal)

        except Exception as e:
            print(f"保存失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalAudioSeparator(root)
    root.geometry("1400x800")
    root.mainloop()