import numpy as np
import torch
from scipy.sparse import diags
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GraphProcessor:
    def __init__(self, adjacency, device='cpu'):
        """
        初始化图处理器。

        Args:
            adjacency (np.ndarray): 邻接矩阵。
            device (str, optional): 计算设备，'cpu'或'cuda'。 Defaults to 'cpu'.
        """
        self.adjacency = adjacency
        self.device = device
        self.transition_matrix = self.get_transition_matrix()

    def get_transition_matrix(self):
        """
        计算概率转移矩阵 P = D^{-1} * A

        Returns:
            torch.Tensor: 概率转移矩阵，移动到指定设备。
        """
        print("计算概率转移矩阵")
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()  # 计算每个节点的度
        degrees[degrees == 0] = 1  # 将度为0的节点度设置为1，避免除以0
        D_inv = diags(1.0 / degrees)  # 构建度的逆对角矩阵
        P = D_inv.dot(self.adjacency)  # 计算概率转移矩阵 P = D^{-1} * A

        # 检查 P 是否有 toarray 方法（即是否为稀疏矩阵）
        if hasattr(P, 'toarray'):
            P = P.toarray()
        return torch.tensor(P, dtype=torch.float32).to(self.device)  # 转换为PyTorch张量并移动到设备

    def personalized_random_walk1_convergence(self, alpha=0.4, max_iter=500, record_every=1):
        """
        计算亲和矩阵并记录收敛差异，策略1：从二阶概率转移矩阵开始累加。

        Args:
            alpha (float, optional): 重启概率。 Defaults to 0.4.
            max_iter (int, optional): 最大迭代次数。 Defaults to 500.
            record_every (int, optional): 记录差异的频率。 Defaults to 1.

        Returns:
            tuple: 最终亲和矩阵 M，差异列表 differences，以及记录的亲和矩阵列表 M_records。
        """
        P = self.transition_matrix
        M = torch.zeros_like(P).to(self.device)
        power_P = torch.eye(P.shape[0], device=self.device)  # P^0 = I
        power_P = power_P.matmul(P)  # P^1
        power_P = power_P.matmul(P)  # P^2

        differences = []
        M_records = []
        last_M = None

        for k in tqdm(range(max_iter), desc="亲和矩阵计算中 (策略1)"):
            if alpha == 0:
                M += power_P  # 当 alpha = 0 时，M 为所有 P^k 的累加，K从2开始
            else:
                M += alpha * (1 - alpha) ** (k + 1) * power_P  # 计算亲和矩阵 M
            power_P = power_P.matmul(P)  # 计算 P^{k+2}

            if (k + 1) % record_every == 0:
                # 记录当前未归一化的亲和矩阵
                M_records.append(M.clone())

                # 计算与上一次记录的矩阵差异（未归一化）
                if last_M is not None:
                    diff = torch.norm(M - last_M, p='fro').item()
                    differences.append(diff)
                last_M = M.clone()

        # 返回最终亲和矩阵、差异列表和记录的亲和矩阵列表
        return M, differences, M_records

def generate_random_adjacency(num_nodes=10, connection_prob=0.3, seed=42):
    """
    生成一个随机对称邻接矩阵。

    Args:
        num_nodes (int, optional): 节点数量。 Defaults to 10.
        connection_prob (float, optional): 节点间连接概率。 Defaults to 0.3.
        seed (int, optional): 随机种子。 Defaults to 42.

    Returns:
        np.ndarray: 生成的邻接矩阵。
    """
    np.random.seed(seed)
    adjacency = np.random.rand(num_nodes, num_nodes) < connection_prob
    adjacency = adjacency.astype(float)
    np.fill_diagonal(adjacency, 0)  # 去除自环
    adjacency = (adjacency + adjacency.T) / 2  # 对称化
    adjacency[adjacency > 1] = 1  # 避免超过1的值
    return adjacency

def main():
    # 1. 生成随机邻接矩阵
    num_nodes = 10
    connection_prob = 0.3
    adjacency = generate_random_adjacency(num_nodes=num_nodes, connection_prob=connection_prob)
    print("随机生成的邻接矩阵：\n", adjacency)

    # 2. 初始化图处理器
    device = 'cpu'  # 如果有GPU，可以设置为 'cuda'
    processor = GraphProcessor(adjacency, device=device)

    # 3. 计算亲和矩阵并记录收敛差异
    alpha = 0.4
    max_iter = 500
    record_every = 1  # 每1次迭代记录一次

    print("\n计算策略1的亲和矩阵...")
    M1, diffs1, M_records = processor.personalized_random_walk1_convergence(alpha=alpha, max_iter=max_iter, record_every=record_every)

    # 4. 可视化收敛差异（添加放大效果）
    iterations_diff = range(2, max_iter + 1, record_every)
    differences_plot = diffs1

    # 创建主图和放大图
    fig = plt.figure(figsize=(12, 6))
    
    # 主图
    ax1 = plt.subplot(111)
    ax1.plot(range(1, len(diffs1) + 1), differences_plot, marker='o', color='blue')
    ax1.set_title('Convergence of Strategy 1 Affinity Matrix (500 Iterations)')
    ax1.set_xlabel('Number of Records')
    ax1.set_ylabel('Matrix Difference (Frobenius Norm)')
    ax1.grid(True)

    # 创建放大图
    # 在主图中标记要放大的区域
    x1, x2 = 0, 30
    y1 = -0.01
    y2 = 0.03
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linestyle='--')
    ax1.add_patch(rect)

    # 创建放大子图，调整位置和大小
    axins = ax1.inset_axes([0.35, 0.35, 0.55, 0.55])
    axins.plot(range(1, 31), differences_plot[:30], marker='o', color='blue')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True)

    # 连接主图和放大图
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="red", linestyle='--')

    plt.tight_layout()
    
    # 保存为PDF格式
    plt.savefig('convergence_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. 可视化未归一化亲和矩阵的收敛情况
    selected_pairs = [(0, 4), (1, 3), (2, 6), (5, 8)]

    plt.figure(figsize=(10, 6))

    for (i, j) in selected_pairs:
        values = [M_record[i, j].item() for M_record in M_records]
        plt.plot(range(1, len(values) + 1), values, marker='o', label=f'Node {i} to Node {j}')

    plt.title('Convergence of Non-normalized Affinity Matrix')
    plt.xlabel('Number of Records')
    plt.ylabel('Affinity Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 同样保存为PDF格式
    plt.savefig('affinity_values_convergence.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. 创建动画展示亲和矩阵的收敛过程
    record_every_anim = 1
    print("\nRecalculating Strategy 1 affinity matrix for animation...")
    _, _, M_records_anim = processor.personalized_random_walk1_convergence(alpha=alpha, max_iter=max_iter, record_every=record_every_anim)

    # 创建动画
    fig, ax = plt.subplots(figsize=(6, 5))

    def update(frame):
        ax.clear()
        M_current = M_records_anim[frame]
        im = ax.imshow(M_current.cpu().numpy(), cmap='viridis', vmin=0, vmax=np.max(M1.cpu().numpy()))
        ax.set_title(f'Affinity Matrix Convergence (Iteration {frame + 1})')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im,

    anim = FuncAnimation(fig, update, frames=len(M_records_anim), interval=50, blit=False)

    plt.close(fig)  # Prevents additional static plot from showing

    # 显示动画
    from IPython.display import HTML
    # 如果在Jupyter Notebook中运行，可以使用以下方式显示动画
    # HTML(anim.to_jshtml())

    # 如果在本地运行脚本，可以保存动画为GIF或MP4
    anim.save('affinity_matrix_convergence.gif', writer='imagemagick', fps=30)
    print("\n动画已保存为 'affinity_matrix_convergence.gif'。")

    # 7. 打印最终亲和矩阵
    print("\n策略1计算的亲和矩阵 M1（未归一化）：\n", M1.cpu().numpy())

if __name__ == "__main__":
    main()
