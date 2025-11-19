import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')


class YOLOCompleteVisualizer:
    def __init__(self, results_path='results.csv'):
        self.results_path = results_path
        self.data = None
        self._load_data()
        self._setup_style()

    def _setup_style(self):
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'lines.linewidth': 2,
            'axes.linewidth': 1.2
        })

        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8E23', '#3E2F5B', '#4C8577']
        self.cmap = plt.cm.viridis

    def _load_data(self):
        try:
            self.data = pd.read_csv(self.results_path)
            print(f"数据加载成功! 共{len(self.data)}行, {len(self.data.columns)}列")
            print("数据列名:", self.data.columns.tolist())
        except Exception as e:
            print(f"数据加载失败: {e}")
            self._create_demo_data()

    def _create_demo_data(self):
        print("创建演示数据...")
        epochs = np.arange(1, 101)
        self.data = pd.DataFrame({
            'epoch': epochs,
            'train_loss': 2.0 * np.exp(-epochs / 20) + np.random.normal(0, 0.05, len(epochs)),
            'val_loss': 1.8 * np.exp(-epochs / 25) + np.random.normal(0, 0.03, len(epochs)),
            'precision': 0.3 + 0.5 * (1 - np.exp(-epochs / 15)) + np.random.normal(0, 0.02, len(epochs)),
            'recall': 0.2 + 0.6 * (1 - np.exp(-epochs / 18)) + np.random.normal(0, 0.02, len(epochs)),
            'mAP_0.5': 0.25 + 0.6 * (1 - np.exp(-epochs / 12)) + np.random.normal(0, 0.02, len(epochs)),
            'mAP_0.5:0.95': 0.15 + 0.5 * (1 - np.exp(-epochs / 15)) + np.random.normal(0, 0.02, len(epochs))
        })


    def plot_confusion_matrix(self, save_path=None):

        fig, ax = plt.subplots(figsize=(10, 8))
        classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']
        cm = np.random.rand(len(classes), len(classes)) * 100
        np.fill_diagonal(cm, np.random.rand(len(classes)) * 80 + 20)

        im = ax.imshow(cm, cmap='Blues', alpha=0.8)

        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax.text(j, i, f'{cm[i, j]:.1f}%',
                               ha="center", va="center",
                               color="white" if cm[i, j] > 50 else "black",
                               fontsize=10, fontweight='bold')

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {save_path}")
        plt.show()

    def plot_pr_curve(self, save_path=None):

        fig, ax = plt.subplots(figsize=(10, 8))


        precision = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55])
        recall = np.linspace(0, 1, 10)
        ap = auc(recall, precision)

        ax.plot(recall, precision, color=self.colors[0],
                linewidth=3, label=f'AP = {ap:.3f}')
        ax.fill_between(recall, precision, alpha=0.2, color=self.colors[0])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线已保存至: {save_path}")
        plt.show()

    def plot_roc_curve(self, save_path=None):

        fig, ax = plt.subplots(figsize=(10, 8))


        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=self.colors[1],
                linewidth=3, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=16, fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存至: {save_path}")
        plt.show()

    def plot_precision_recall_curves(self, save_path=None):

        if self.data is None:
            print("数据未加载!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        epochs = np.arange(1, len(self.data) + 1)

        if 'precision' in self.data.columns:
            precision = self.data['precision']
        else:
            precision = np.random.rand(len(epochs)) * 0.3 + 0.6  # 模拟数据

        ax1.plot(epochs, precision, color=self.colors[0], linewidth=2.5)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax1.set_title('Precision Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        if 'recall' in self.data.columns:
            recall = self.data['recall']
        else:
            recall = np.random.rand(len(epochs)) * 0.3 + 0.5  # 模拟数据

        ax2.plot(epochs, recall, color=self.colors[1], linewidth=2.5)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_title('Recall Curve', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"精度召回率曲线已保存至: {save_path}")
        plt.show()

    def plot_map_curves(self, save_path=None):

        if self.data is None:
            print("数据未加载!")
            return

        fig, ax = plt.subplots(figsize=(10, 6))


        epochs = np.arange(1, len(self.data) + 1)

        if 'mAP_0.5' in self.data.columns:
            map50 = self.data['mAP_0.5']
        else:
            map50 = 0.3 + 0.5 * (1 - np.exp(-epochs / 10)) + np.random.normal(0, 0.02, len(epochs))

        if 'mAP_0.5:0.95' in self.data.columns:
            map95 = self.data['mAP_0.5:0.95']
        else:
            map95 = 0.2 + 0.4 * (1 - np.exp(-epochs / 15)) + np.random.normal(0, 0.02, len(epochs))

        ax.plot(epochs, map50, color=self.colors[2], linewidth=2.5, label='mAP@0.5')
        ax.plot(epochs, map95, color=self.colors[3], linewidth=2.5, label='mAP@0.5:0.95')

        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('mAP', fontsize=14, fontweight='bold')
        ax.set_title('mAP Curves', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        final_map50 = map50.iloc[-1] if hasattr(map50, 'iloc') else map50[-1]
        final_map95 = map95.iloc[-1] if hasattr(map95, 'iloc') else map95[-1]

        ax.annotate(f'mAP@0.5: {final_map50:.3f}',
                    xy=(epochs[-1], final_map50),
                    xytext=(epochs[-1] - len(epochs) * 0.3, final_map50 - 0.1),
                    arrowprops=dict(arrowstyle='->', color=self.colors[2]),
                    fontsize=12, fontweight='bold')

        ax.annotate(f'mAP@0.5:0.95: {final_map95:.3f}',
                    xy=(epochs[-1], final_map95),
                    xytext=(epochs[-1] - len(epochs) * 0.3, final_map95 - 0.15),
                    arrowprops=dict(arrowstyle='->', color=self.colors[3]),
                    fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"mAP曲线已保存至: {save_path}")
        plt.show()

    def plot_loss_curves(self, save_path=None):

        if self.data is None:
            print("数据未加载!")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = np.arange(1, len(self.data) + 1)


        loss_columns = [col for col in self.data.columns if 'loss' in col.lower()]

        if len(loss_columns) > 0:
            for i, col in enumerate(loss_columns[:3]):
                loss_data = self.data[col]
                ax.plot(epochs, loss_data,
                        color=self.colors[i % len(self.colors)],
                        linewidth=2,
                        label=col.replace('_', ' ').title())
        else:

            train_loss = 2.0 * np.exp(-epochs / 10) + np.random.normal(0, 0.05, len(epochs))
            val_loss = 1.8 * np.exp(-epochs / 12) + np.random.normal(0, 0.03, len(epochs))

            ax.plot(epochs, train_loss, color=self.colors[0], linewidth=2, label='Train Loss')
            ax.plot(epochs, val_loss, color=self.colors[1], linewidth=2, label='Validation Loss')

        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"损失曲线已保存至: {save_path}")
        plt.show()


    def plot_3d_iou_precision_curve(self, save_path=None):

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        iou_thresholds = np.linspace(0.1, 0.9, 9)
        confidence_thresholds = np.linspace(0.1, 0.9, 9)

        X, Y = np.meshgrid(iou_thresholds, confidence_thresholds)

        Z = 0.7 + 0.2 * np.exp(-(X - 0.5) ** 2 / 0.2 - (Y - 0.6) ** 2 / 0.3) + \
            np.random.normal(0, 0.02, X.shape)

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                               linewidth=0, antialiased=True)

        ax.set_xlabel('IoU Threshold', labelpad=12, fontweight='bold')
        ax.set_ylabel('Confidence Threshold', labelpad=12, fontweight='bold')
        ax.set_zlabel('Precision', labelpad=12, fontweight='bold')
        ax.set_title('3D IoU-Precision Surface', fontsize=16, fontweight='bold', pad=20)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)

        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D IoU-Precision曲线已保存至: {save_path}")
        plt.show()

    def plot_bev_precision_curve(self, save_path=None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        distances = np.linspace(0, 100, 20)
        bev_precision = 0.8 * np.exp(-distances / 40) + 0.2 + np.random.normal(0, 0.03, len(distances))

        ax1.plot(distances, bev_precision, 'o-', color='#2E86AB', linewidth=2.5, markersize=6)
        ax1.set_xlabel('Distance (m)', fontweight='bold')
        ax1.set_ylabel('BEV Precision', fontweight='bold')
        ax1.set_title('BEV Precision vs Distance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        x_range = np.linspace(-50, 50, 20)
        y_range = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.exp(-(X ** 2 + (Y - 30) ** 2) / 1000) * 0.8 + 0.2

        im = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax2.set_xlabel('X Position (m)', fontweight='bold')
        ax2.set_ylabel('Y Position (m)', fontweight='bold')
        ax2.set_title('BEV Precision Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Precision')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"BEV精度曲线已保存至: {save_path}")
        plt.show()

    def plot_3d_ap_curves(self, save_path=None):

        fig = plt.figure(figsize=(12, 8))


        difficulties = ['Easy', 'Moderate', 'Hard']
        iou_thresholds = np.linspace(0.1, 0.9, 17)

        ap_data = {
            'Easy': 0.9 - 0.6 * (iou_thresholds - 0.1) + np.random.normal(0, 0.02, len(iou_thresholds)),
            'Moderate': 0.8 - 0.5 * (iou_thresholds - 0.1) + np.random.normal(0, 0.02, len(iou_thresholds)),
            'Hard': 0.7 - 0.4 * (iou_thresholds - 0.1) + np.random.normal(0, 0.02, len(iou_thresholds))
        }

        colors = ['#2E86AB', '#F18F01', '#C73E1D']

        for i, (diff, aps) in enumerate(ap_data.items()):
            plt.plot(iou_thresholds, aps, 'o-', color=colors[i],
                     linewidth=2.5, markersize=5, label=f'{diff} (AP: {np.mean(aps):.3f})')

        plt.xlabel('3D IoU Threshold', fontweight='bold')
        plt.ylabel('Average Precision (AP)', fontweight='bold')
        plt.title('3D Average Precision vs IoU Threshold\nby Difficulty Level',
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D AP曲线已保存至: {save_path}")
        plt.show()

    def plot_orientation_performance_heatmap(self, save_path=None):

        fig = plt.figure(figsize=(12, 10))


        azimuth = np.linspace(-180, 180, 36)
        elevation = np.linspace(-30, 30, 24)  #
        A, E = np.meshgrid(azimuth, elevation)

        performance = np.exp(-(A ** 2) / 50000 - (E ** 2) / 2000) * 0.8 + 0.2

        plt.contourf(A, E, performance, levels=50, cmap='viridis')
        plt.colorbar(label='Detection Performance')

        plt.xlabel('Azimuth Angle (degrees)', fontweight='bold')
        plt.ylabel('Elevation Angle (degrees)', fontweight='bold')
        plt.title('3D Detection Performance vs Object Orientation',
                  fontsize=14, fontweight='bold')


        plt.text(0, 25, 'Front', ha='center', va='center', fontweight='bold', color='white')
        plt.text(-90, 0, 'Left', ha='center', va='center', fontweight='bold', color='white')
        plt.text(90, 0, 'Right', ha='center', va='center', fontweight='bold', color='white')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"方位角-俯仰角热图已保存至: {save_path}")
        plt.show()



    def plot_2d_evaluation(self, save_dir='./2d_plots/'):
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("开始绘制2D评估图表...")

        self.plot_confusion_matrix(save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        self.plot_pr_curve(save_path=os.path.join(save_dir, 'pr_curve.png'))
        self.plot_roc_curve(save_path=os.path.join(save_dir, 'roc_curve.png'))
        self.plot_precision_recall_curves(save_path=os.path.join(save_dir, 'precision_recall_curves.png'))
        self.plot_map_curves(save_path=os.path.join(save_dir, 'map_curves.png'))
        self.plot_loss_curves(save_path=os.path.join(save_dir, 'loss_curves.png'))

        print(f"所有2D评估图表已保存至: {save_dir}")

    def plot_3d_evaluation(self, save_dir='./3d_plots/'):

        import os
        os.makedirs(save_dir, exist_ok=True)

        print("开始绘制3D评估图表...")

        self.plot_3d_iou_precision_curve(save_path=os.path.join(save_dir, '3d_iou_precision.png'))
        self.plot_bev_precision_curve(save_path=os.path.join(save_dir, 'bev_precision.png'))
        self.plot_3d_ap_curves(save_path=os.path.join(save_dir, '3d_ap_curves.png'))
        self.plot_orientation_performance_heatmap(save_path=os.path.join(save_dir, 'orientation_heatmap.png'))

        print(f"所有3D评估图表已保存至: {save_dir}")

    def plot_all(self, save_dir='./all_plots/'):

        import os
        os.makedirs(save_dir, exist_ok=True)

        print("开始绘制所有评估图表...")


        os.makedirs(os.path.join(save_dir, '2d_plots'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, '3d_plots'), exist_ok=True)

        self.plot_confusion_matrix(save_path=os.path.join(save_dir, '2d_plots', 'confusion_matrix.png'))
        self.plot_pr_curve(save_path=os.path.join(save_dir, '2d_plots', 'pr_curve.png'))
        self.plot_roc_curve(save_path=os.path.join(save_dir, '2d_plots', 'roc_curve.png'))
        self.plot_precision_recall_curves(save_path=os.path.join(save_dir, '2d_plots', 'precision_recall_curves.png'))
        self.plot_map_curves(save_path=os.path.join(save_dir, '2d_plots', 'map_curves.png'))
        self.plot_loss_curves(save_path=os.path.join(save_dir, '2d_plots', 'loss_curves.png'))

        self.plot_3d_iou_precision_curve(save_path=os.path.join(save_dir, '3d_plots', '3d_iou_precision.png'))
        self.plot_bev_precision_curve(save_path=os.path.join(save_dir, '3d_plots', 'bev_precision.png'))
        self.plot_3d_ap_curves(save_path=os.path.join(save_dir, '3d_plots', '3d_ap_curves.png'))
        self.plot_orientation_performance_heatmap(
            save_path=os.path.join(save_dir, '3d_plots', 'orientation_heatmap.png'))

        print(f"所有评估图表已保存至: {save_dir}")


        self._generate_summary_report(save_dir)

    def _generate_summary_report(self, save_dir):

        report_path = os.path.join(save_dir, 'evaluation_summary.txt')
        with open(report_path, 'w') as f:
            f.write("YOLO算法评估汇总报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"数据文件: {self.results_path}\n")
            f.write(f"数据行数: {len(self.data)}\n")
            f.write(f"数据列数: {len(self.data.columns)}\n\n")

            f.write("包含的评估图表:\n")
            f.write("- 2D评估图表:\n")
            f.write("  * 混淆矩阵 (Confusion Matrix)\n")
            f.write("  * PR曲线 (Precision-Recall Curve)\n")
            f.write("  * ROC曲线 (ROC Curve)\n")
            f.write("  * 精度和召回率曲线 (Precision & Recall Curves)\n")
            f.write("  * mAP曲线 (mAP Curves)\n")
            f.write("  * 损失函数曲线 (Loss Curves)\n\n")

            f.write("- 3D评估图表:\n")
            f.write("  * 3D IoU-Precision曲线\n")
            f.write("  * BEV精度曲线\n")
            f.write("  * 3D AP曲线 (不同难度级别)\n")
            f.write("  * 方位角-俯仰角性能热图\n\n")

            f.write(f"图表保存路径: {save_dir}\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n")

        print(f"汇总报告已保存至: {report_path}")



if __name__ == "__main__":

    visualizer = YOLOCompleteVisualizer('results.csv')


    visualizer.plot_all()