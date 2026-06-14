交差エントロピー誤差（Cross-Entropy Loss）について、ICML 2023およびNeurIPS 2018の権威ある論文を中心に、情報理論的基礎から理論的保証、一般化・変種まで体系的に解説します。

***

## 1. 定義と情報理論的意味

**交差エントロピー**は、真の確率分布 $p$ と推定分布 $q$ の間で定義される尺度で、「$q$ に最適化された符号化方式で $p$ からの事象を識別するために必要な平均ビット数」を表します 。 [en.wikipedia](https://en.wikipedia.org/wiki/Cross-entropy)

機械学習では、**交差エントロピー誤差（Cross-Entropy Loss, CE Loss）**として、真のラベル分布 $y$（ワンホットベクトル）とモデルの予測確率分布 $\hat{y}$（Softmax出力）の間のズレを測る損失関数として用いられます：

$$
\mathcal{L}_{\text{CE}}(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

ここで $K$ はクラス数、$\hat{y}_k$ はクラス $k$ の予測確率です 。 [papers.neurips](http://papers.neurips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networkswith-noisy-labels.pdf)

***

## 2. エントロピー・KLダイバージェンスとの関係

交差エントロピーは以下のように分解できます ： [tensortonic](https://www.tensortonic.com/ml-math/information-theory/cross-entropy)

$$
H(p, q) = H(p) + D_{\text{KL}}(p \parallel q)
$$

- $H(p)$：真の分布のエントロピー（データ固有の不確実性、定数）
- $D_{\text{KL}}(p \parallel q)$：KLダイバージェンス（$p$ と $q$ の乖離）

**最小化の意味**：$H(p)$ は定数なので、交差エントロピーを最小化することは KLダイバージェンスを最小化することと等価であり、「予測分布 $q$ を真の分布 $p$ に近づける」ことになります 。 [tensortonic](https://www.tensortonic.com/ml-math/information-theory/cross-entropy)

***

## 3. バイナリ分類と多クラス分類での形

| タスク | 損失関数 |
|--------|----------|
| **バイナリ分類** (BCE) | $\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$ |
| **多クラス分類** (CCE) | $\mathcal{L} = -\sum_{k=1}^K y_k \log \hat{y}_k$ |

多クラスでは、正解クラス $c$ のみ $y_c=1$ なので、実質 $\mathcal{L} = -\log \hat{y}_c$ となります 。 [disassemble-channel](https://disassemble-channel.com/cross-entropy-loss/)

**記号の意味：**

| 記号 | バイナリ分類 (BCE) | 多クラス分類 (CCE) |
|------|-------------------|-------------------|
| $y$ | 正解ラベル（0 or 1のスカラー） | 正解ラベルのone-hotベクトル |
| $y_k$ | — | $y$ の第 $k$ 成分（正解クラスのみ1、他は0） |
| $\hat{y}$ | Sigmoid出力（陽性クラスの予測確率） | Softmax出力ベクトル |
| $\hat{y}_k$ | — | $\hat{y}$ の第 $k$ 成分（クラス $k$ の予測確率） |

例（BCE、正解＝猫(1)、予測確率0.8）：$y=1$、$\hat{y}=0.8$ → $\mathcal{L} = -[1 \cdot \log 0.8 + 0 \cdot \log 0.2] = -\log 0.8$

例（CCE、3クラス分類、正解＝猫 $k=2$）：$y=[0,1,0]$、$\hat{y}=[0.1, 0.8, 0.1]$ → $\mathcal{L} = -\log 0.8$

***

## 4. Softmaxとの組み合わせ・勾配の性質

ニューラルネットワークでは **Softmax出力層** と組み合わせて使われます。このとき、ロジット $z_k$ から予測確率 $\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$ となり、交差エントロピーは **ロジスティック損失（多項ロジット損失）** と一致します 。 [arxiv](https://arxiv.org/abs/2304.07288)

### 勾配の重要な性質

パラメータ $\theta$ に関する勾配 ： [papers.neurips](http://papers.neurips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networkswith-noisy-labels.pdf)

$$
\frac{\partial \mathcal{L}}{\partial \theta} = -\frac{1}{\hat{y}_c} \frac{\partial \hat{y}_c}{\partial \theta} \quad \text{(正解クラス $c$ のみ)}
$$

**重要な含意**：
- **難しいサンプル（予測確率 $\hat{y}_c$ が小さい）ほど大きな勾配**を受け取る → 暗黙的な重み付けで困難例に注力
- これはクリーンなデータでは望ましいが、**ラベルノイズがあると誤ラベルに過学習**する原因になる [papers.neurips](http://papers.neurips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networkswith-noisy-labels.pdf)

***

## 5. 理論的保証：H-一貫性境界（ICML 2023）

Mao, Mohri, Zhong (2023)  は、交差エントロピー（ロジスティック損失）を含む **comp-sum損失族** に対して、初めて **H-一貫性境界** を導出しました。 [arxiv](https://arxiv.org/abs/2304.07288)

### 定理（簡略版）
対称かつ完全な仮説集合 $H$ に対し、任意の $h \in H$ で：

$$
R_{0-1}(h) - R^*_{0-1}(H) \le \Gamma_1\bigl(R_{\text{CE}}(h) - R^*_{\text{CE}}(H) + M_{\text{CE}}(H)\bigr) - M_{0-1}(H)
$$

ここで：
- $R_{0-1}$：ゼロワン誤差（真の分類誤差）
- $R_{\text{CE}}$：交差エントロピー誤差
- $M$：**最小化可能性ギャップ**（仮説集合 $H$ の制約によるギャップ）
- $\Gamma_1(t) = \frac{1+t}{2}\log(1+t) + \frac{1-t}{2}\log(1-t)$ の逆関数

**主要な知見**：
1. **非漸近的・仮説集合依存**の保証（ベイズ一貫性より強い）
2. **関数形は平方根型**：$\sqrt{R_{\text{CE}} - R^*_{\text{CE}}}$ に比例してゼロワン誤差が抑制される
3. **バウンドはタイト**（改善不可能） [arxiv](https://arxiv.org/abs/2304.07288)

### 最小化可能性ギャップの比較
comp-sum族では $\tau$ パラメータで以下を統一的に扱える ： [arxiv](https://arxiv.org/abs/2304.07288)

| $\tau$ | 損失関数 | 最小化可能性ギャップ | クラス数 $n$ への依存 |
|--------|----------|---------------------|----------------------|
| 0 | Sum-Exponential | 最大 | なし |
| 1 | **交差エントロピー（ロジスティック）** | 中程度 | なし |
| 1<τ<2 | 一般化交差エントロピー | 小さい | $\sqrt{n}$ |
| 2 | MAE | 最小 | $n$ |

交差エントロピー（$\tau=1$）は、**ギャップの大きさ**と**クラス数依存のなさ**のバランスが良く、実用的にも最も優れることが理論・実験両面で示されています 。 [arxiv](https://arxiv.org/abs/2304.07288)

***

## 6. 一般化交差エントロピー（GCE）とノイズロバスト性（NeurIPS 2018）

Zhang & Sabuncu (2018)  は、交差エントロピーのノイズ感受性を改善する **$L_q$ 損失** を提案： [papers.neurips](http://papers.neurips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networkswith-noisy-labels.pdf)

$$
\mathcal{L}_q(f(x), e_j) = \frac{1 - f_j(x)^q}{q}, \quad q \in (0, 1]
$$

### 性質
- $q \to 0$ で **交差エントロピー** と一致（L'Hôpitalの定理）
- $q = 1$ で **MAE / Unhinged Loss** と一致
- **勾配**：$-\hat{y}_c^{q-1} \nabla_\theta \hat{y}_c$
  - $\hat{y}_c^q$ の重みで「難しいサンプルへの注力度」を調整
  - $q$ 大きい → ノイズロバストだが収束遅い
  - $q$ 小さい → 収束早いがノイズに弱い

### 実用的推奨
- **$q = 0.7$** 付近でトレードオフが良好（CIFAR-10/100で検証済み） [papers.neurips](http://papers.neurips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networkswith-noisy-labels.pdf)
- さらに **Truncated $L_q$**（閾値 $k$ 以下の予測は損失を定数に）でノイズロバスト性を強化可能

***

## 7. 対抗的ロバストネスへの拡張

Mao et al. (2023)  は **Smooth Adversarial Comp-Sum Loss** を導入し、交差エントロピー（$\tau=1$）ベースの **ADV-COMP-SUM** アルゴリズムで： [arxiv](https://arxiv.org/abs/2304.07288)
- CIFAR-10/100, SVHN で **TRADESを上回る** 対抗的精度
- **クリーン精度も同時に向上**（従来のトレードオフを突破）

***

## 8. 実装上の注意点（数値安定性）

| ポイント | 対策 |
|----------|------|
| $\log(0)$ 回避 | $\hat{y}_k$ に小さな $\epsilon$ (1e-7等) をクリッピング |
| Log-Softmax + NLL | PyTorchの `F.cross_entropy` / `nn.CrossEntropyLoss` は内部で LogSoftmax + NLLLoss を結合し数値安定 |
| ラベルスムージング | 正解ラベルを $(1-\alpha)\delta_{y_c} + \alpha/K$ に置換し過信抑制 |

***

## 9. まとめ：なぜ交差エントロピーが標準なのか

| 観点 | 理由 |
|------|------|
| **情報理論** | 真分布と予測分布の KLダイバージェンス最小化と等価 |
| **最適化** | Softmaxと組み合わせると凸・勾配が $\hat{y}_c - y_c$ 形で扱いやすい |
| **理論保証** | H-一貫性境界がタイトで、クラス数依存がない唯一の損失（$\tau=1$） |
| **実用性** | 収束速度・精度・実装の容易さのバランスが最良 |
| **拡張性** | GCE・Truncated・Adversarial版など派生手法が豊富 |

***

## 参考文献（主要論文）

1. **Mao, Mohri, Zhong** (2023). *Cross-Entropy Loss Functions: Theoretical Analysis and Applications*. **ICML 2023** (arXiv:2304.07288) — H-一貫性境界、comp-sum族の統一理論、対抗的拡張 [arxiv](https://arxiv.org/abs/2304.07288)
2. **Zhang & Sabuncu** (2018). *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels*. **NeurIPS 2018** (arXiv:1805.07836) — $L_q$ 損失、ノイズロバスト性の理論・実証 [papers.neurips](http://papers.neurips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networkswith-noisy-labels.pdf)
3. **Ghosh et al.** (2017). *Robust Loss Functions under Label Noise for Deep Neural Networks*. AAAI 2017 — MAEのノイズロバスト性証明（GCEの基礎） [papers.neurips](http://papers.neurips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networkswith-noisy-labels.pdf)
4. **Cover & Thomas** (2006). *Elements of Information Theory*. Wiley — 交差エントロピー・KLダイバージェンスの教科書的定義

***

ご自身の研究（SAM + アダプターによるTEM画像セグメンテーション）では、**クラス不均衡**や**アノテーションノイズ**が懸念される場合、GCE ($q \approx 0.7$) やラベルスムージング付き交差エントロピー、あるいは Focal Loss 的な変種の検討も有用かと思います。具体的なデータセット特性に合わせて、どの変種が適そうかさらに議論できます。


---

# 逆伝播
この記事の逆伝播導出を、ご指定の記号（**教師ラベル $t \to \mathbf{y}$、モデル出力 $y \to \hat{\mathbf{y}}$**）に書き換えて、ステップバイステップで追います。

***

## 記号対応表

| 記事の記号 | 新しい記号 | 意味 |
|------------|------------|------|
| $\mathbf{T}$ / $\mathbf{t}_n$ | $\mathbf{y}$ / $\mathbf{y}_n$ | **教師ラベル**（one-hotベクトル） |
| $\mathbf{Y}$ / $\mathbf{y}_n$ | $\hat{\mathbf{y}}$ / $\hat{\mathbf{y}}_n$ | **モデル予測確率**（Softmax出力） |
| $\mathbf{A}$ / $\mathbf{a}_n$ | $\mathbf{z}$ / $\mathbf{z}_n$ | **ロジット**（Softmax入力、Affine出力） |
| $L$ | $L$ | **交差エントロピー損失**（スカラ） |
| $N$ | $N$ | **バッチサイズ** |
| $K$ | $K$ | **クラス数** |

***

## 1. 順伝播の定義（確認）

**単一サンプル $n$ について**：

$$
\begin{aligned}
\hat{y}_{n,k} &= \frac{\exp(z_{n,k})}{\sum_{j=1}^K \exp(z_{n,j})} \quad (\text{Softmax}) \\[4pt]
L_n &= -\sum_{k=1}^K y_{n,k} \log \hat{y}_{n,k} \quad (\text{Cross-Entropy, 1サンプル分の損失})
\end{aligned}
$$

**バッチ全体の損失**（平均）：

$$
L = \frac{1}{N} \sum_{n=1}^N L_n = -\frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \log \hat{y}_{n,k}
$$

***

## 2. 逆伝播の目標

**ロジット $\mathbf{z}_n$ に関する損失の勾配** $\displaystyle \frac{\partial L}{\partial \mathbf{z}_n} \in \mathbb{R}^K$ を求める。

連鎖律で分解：

$$
\frac{\partial L}{\partial z_{n,k}} = \sum_{i=1}^K \frac{\partial L}{\partial \hat{y}_{n,i}} \frac{\partial \hat{y}_{n,i}}{\partial z_{n,k}}
$$

***

## 3. Step 1：$\partial L / \partial \hat{y}_{n,i}$ の計算

$$
L = -\frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \log \hat{y}_{n,k}
$$

$\hat{y}_{n,i}$ は $n$ 番目のサンプル・$i$ 番目のクラスの予測確率。他のサンプル・他のクラスの $\hat{y}$ には依存しないので：

$$
\frac{\partial L}{\partial \hat{y}_{n,i}} = -\frac{1}{N} \frac{y_{n,i}}{\hat{y}_{n,i}}
$$

***

## 4. Step 2：$\partial \hat{y}_{n,i} / \partial z_{n,k}$ の計算（Softmaxのヤコビアン）

Softmaxの微分はよく知られた形：

$$
\frac{\partial \hat{y}_{n,i}}{\partial z_{n,k}} = 
\begin{cases}
\hat{y}_{n,i}(1 - \hat{y}_{n,i}) & (i = k) \\[4pt]
-\hat{y}_{n,i} \hat{y}_{n,k} & (i \ne k)
\end{cases}
$$

コンパクトに書くと：

$$
\frac{\partial \hat{y}_{n,i}}{\partial z_{n,k}} = \hat{y}_{n,i} (\delta_{ik} - \hat{y}_{n,k})
$$

where $\delta_{ik}$ はクロネッカーのデルタ。

***

## 5. Step 3：連鎖律で合成

$$
\begin{aligned}
\frac{\partial L}{\partial z_{n,k}}
&= \sum_{i=1}^K \left( -\frac{1}{N} \frac{y_{n,i}}{\hat{y}_{n,i}} \right) \cdot \hat{y}_{n,i} (\delta_{ik} - \hat{y}_{n,k}) \\[4pt]
&= -\frac{1}{N} \sum_{i=1}^K y_{n,i} (\delta_{ik} - \hat{y}_{n,k}) \\[4pt]
&= -\frac{1}{N} \left( \sum_{i=1}^K y_{n,i} \delta_{ik} - \sum_{i=1}^K y_{n,i} \hat{y}_{n,k} \right) \\[4pt]
&= -\frac{1}{N} \left( y_{n,k} - \hat{y}_{n,k} \underbrace{\sum_{i=1}^K y_{n,i}}_{=1} \right) \quad \text{（$\mathbf{y}_n$ は one-hot なので総和=1）} \\[4pt]
&= \frac{1}{N} (\hat{y}_{n,k} - y_{n,k})
\end{aligned}
$$

***

## 6. 最終結果：ベクトル・行列形

**単一サンプル**：
$$
\boxed{\frac{\partial L}{\partial \mathbf{z}_n} = \frac{1}{N} (\hat{\mathbf{y}}_n - \mathbf{y}_n)}
$$

**バッチ全体**（行列 $\mathbf{Z}, \hat{\mathbf{Y}}, \mathbf{Y} \in \mathbb{R}^{N \times K}$）：
$$
\boxed{\frac{\partial L}{\partial \mathbf{Z}} = \frac{1}{N} (\hat{\mathbf{Y}} - \mathbf{Y})}
$$

***

## 7. 記事のコードとの対応

```python
# 記事のコード（記号を新表記に合わせてコメント）
def backward(self, dout=1):
    batch_size = self.t.shape[0]          # N
    dx = (self.y - self.t) / batch_size   # (ŷ - y) / N
    return dx
```

| 記事の変数 | 新表記 | 内容 |
|------------|--------|------|
| `self.t` | `self.y` | 教師ラベル $\mathbf{Y} \in \mathbb{R}^{N \times K}$ (one-hot) |
| `self.y` | `self.y_hat` | Softmax出力 $\hat{\mathbf{Y}} \in \mathbb{R}^{N \times K}$ |
| `dx` | `dz` | ロジットへの勾配 $\partial L / \partial \mathbf{Z}$ |
| `batch_size` | `N` | バッチサイズ |

***

## 8. 幾何学的・直感的理解

- **$\hat{\mathbf{y}}_n - \mathbf{y}_n$**：予測確率分布と正解分布の**差分ベクトル**
- 正解クラス $c$（$y_{n,c}=1$）では：$\hat{y}_{n,c} - 1 < 0$ → ロジット $z_{n,c}$ を**増やす**方向の勾配
- 非正解クラス $k \ne c$（$y_{n,k}=0$）では：$\hat{y}_{n,k} - 0 > 0$ → ロジット $z_{n,k}$ を**減らす**方向の勾配
- **Softmaxの性質**（総和=1）により、正解クラスを上げると自動的に他クラスが下がる＝整合的な更新

***

## 9. 記事の可視化（2クラス・$N=1$ の場合）の読み替え

記事では $t = ^\top$（クラス0が正解）で可視化しています。新表記では： [qiita](https://qiita.com/kenta1984/items/59a9ef1788e6934fd962)

$$
\mathbf{y} = \begin{bmatrix}1 \\ 0\end{bmatrix}, \quad
\hat{\mathbf{y}} = \begin{bmatrix}\hat{y}_0 \\ \hat{y}_1\end{bmatrix}, \quad
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y} = \begin{bmatrix}\hat{y}_0 - 1 \\ \hat{y}_1\end{bmatrix}
$$

- $x_0 > x_1 \Rightarrow \hat{y}_0 \approx 1, \hat{y}_1 \approx 0 \Rightarrow \text{勾配} \approx \mathbf{0}$（最小損失・学習完了）
- $x_0 < x_1 \Rightarrow \hat{y}_0 \approx 0, \hat{y}_1 \approx 1 \Rightarrow \text{勾配} \approx [-1, 1]^\top$（$x_0$を増やし、$x_1$を減らす強い勾配）

3Dベクトル図の矢印は**損失関数の等高線に直交して「正解方向」を向く**ことが視覚的に確認できます。

***

## まとめ

| ステップ | 式（新表記） |
|----------|-------------|
| 損失 | $L = -\frac{1}{N}\sum_n \sum_k y_{n,k} \log \hat{y}_{n,k}$ |
| $\partial L/\partial \hat{y}$ | $-\frac{1}{N} \mathbf{y} \oslash \hat{\mathbf{y}}$ |
| Softmaxヤコビアン | $\partial \hat{y}_i / \partial z_k = \hat{y}_i(\delta_{ik} - \hat{y}_k)$ |
| **最終勾配** | $\boxed{\frac{\partial L}{\partial \mathbf{Z}} = \frac{1}{N}(\hat{\mathbf{Y}} - \mathbf{Y})}$ |

この「予測 − 正解」という極めてシンプルな形になるのは、**Softmax + Cross-Entropy** の組み合わせが「対数尤度の負」＝「多項ロジット損失」と一致するためです。この美しい性質が、分類タスクでこの組み合わせがデファクトスタンダードである理由の一つです。
