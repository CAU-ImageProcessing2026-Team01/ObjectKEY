# Affine transformation and least-squares fitting

[Back to README](../README.md#how-it-works)

An affine model describes translation, rotation, scale and shear between matched image points.

## Model

For reference point $(x,y)$ and query point $(u,v)$,

$$
\begin{bmatrix}u\\v\end{bmatrix}
=
\begin{bmatrix}a&b\\c&d\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}
+
\begin{bmatrix}t_x\\t_y\end{bmatrix}.
$$

Three non-collinear correspondences determine the six parameters. Additional pairs allow a least-squares estimate that reduces measurement noise.

## Least-squares formulation

Collect correspondences into matrices:

$$
X=
\begin{bmatrix}
x_1&y_1&1\\
\vdots&\vdots&\vdots\\
x_n&y_n&1
\end{bmatrix},\quad
B=
\begin{bmatrix}
a&c\\b&d\\t_x&t_y
\end{bmatrix},\quad
Y=
\begin{bmatrix}
u_1&v_1\\
\vdots&\vdots\\
u_n&v_n
\end{bmatrix}.
$$

The fitted parameters solve

$$
B^*=\operatorname*{arg\,min}_B\lVert XB-Y\rVert_F^2.
$$

Before fitting, the implementation centers each point set and divides by its RMS radius. This improves numerical conditioning; the transformation is converted back to the original coordinates afterward.

## Householder QR implementation

A Householder reflection eliminates entries below a design-matrix pivot:

$$
H=I-2\frac{vv^{T}}{v^{T}v}.
$$

Applying successive reflections produces the QR factorization. The same reflections are applied to both columns of $Y$, leaving an upper-triangular system:

$$
X=QR,\qquad RB=Q^{T}Y.
$$

C++ constructs the reflections and solves the triangular systems by back-substitution. It rejects rank-deficient inputs and invalid transforms. [RANSAC](ransac.md) uses this solver for both minimal hypotheses and all-inlier refinement.

**Code:** `fitAffine` in [Affine.cpp](../src/vision/Affine.cpp).

**References:** [8](../README.md#ref-8), [9](../README.md#ref-9), [10](../README.md#ref-10).
