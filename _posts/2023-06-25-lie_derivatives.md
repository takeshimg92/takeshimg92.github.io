---
layout: post
title: "Lie derivatives and cluttered notation"
author: "Alessandro Morita"
categories: posts
tags: [differential geometry]
image: wald.jpg
---

I've always had a love-hate relationship with notation in mathematics. 

On the one hand, I am an intuitive person who needs to understand the big picture of something before jumping into the details. In this sense, I  prefer my math to be uncluttered and focusing on the key steps of a derivation.

On the other hand, I also need my understanding to be precise to feel that I *really* understand something. That means that the whole procedure must be logical, and everything somehow needs to come together in a meaningful structure. Because of that, I sometimes prefer a more cluttered -- but more precise -- notation.

Almost 10 years ago, I started learning about [Lie derivatives](https://en.wikipedia.org/wiki/Lie_derivative), a beautiful concept in differential geometry that generalizes the notion of taking a small variation of a quantity along a path. It is the key behind concepts like [Killing vectors](https://en.wikipedia.org/wiki/Killing_vector_field) which make the notion of symmetries in physical systems more precise. My first contact with Lie derivatives was through [Wald's General Relativity book](https://www.amazon.com/General-Relativity-Robert-M-Wald/dp/0226870332) and Stewart's [Advanced General Relativity](https://www.amazon.com.br/Advanced-General-Relativity-John-Stewart/dp/0521323193).

Being much less enlightened than both Wald and Stewart, I always had some difficulties with two choices they took:

1. The definition of the Lie derivative is clear, but it takes quite a bit of mental effort to connect all dots and see how the definition makes sense; I would love for the definition to be more transparent, almost algorithmic;

2. The proof of the main result, namely that the Lie derivative of a vector along another equals their commutator, always involves choosing one of them as a coordinate basis vector; I would love a more "general", albeit more involved version, to work without this requirement.

Hence, I am posting here for any other students of differential geometry or general relativity my own proof which is, cluttered as can be, very transparent (for me, of courseon what is going on.

## Set-up: pushforwards and pullbacks

Below, $\phi_t$ is a 1-parameter family of diffeomorphisms induced by a vector field $X$, also called $X$'s orbits; this means that $\phi_t$ is the solution to the initial value problem below, where $x^a$ is local coordinate chart centered around a point $p$:

$$\begin{cases}
\displaystyle \frac{d x^a(t)}{dt} &= X^a(x(t))\\
x^a(0) &= 0
\end{cases}$$

where $X^a(x(t))$ is the value of the components of the vector field $X$ at point $x(t)$.

Provided $X$ is smooth, the solution exists and is unique at least on a neighborhood of $p = \phi_t(0)$; we will assume to be on such a region during the whole discussion below.

First, we will present the rules that allow for the quick rule-of-thumb 

$$\boxed{\mbox{(pushforward of $T$)(stuff) = $T$(pullback of stuff)}},$$ 

where $T$ is some tensor.
> It is crucial here that we deal with diffeomorphisms (which are, in particular, *invertible*); otherwise we could not take inverses and this argument wouldn't work.


### Pullbacks of functions

Given a map $\phi_t$, define a function's *pullback* as 

$$\phi_{t*}f:=f\circ\phi_t.$$

This means that the pullback applied to point $P$ will be equivalent to $f$ applied to $\phi_t(P)$. 
> Pullback of function = function composition.

### Pushforward of vectors

Let $X \in T_p M$. We define a new vector $\phi_t^\ast X \in T_{\phi_t(p)}M$, its *pushforward*, via its action on functions:

$$(\phi_t^*X)(f):= X(f\circ \phi_t) = X(\phi_{t*}f).$$

That is, the pushforward of a vector, applied to a function, is the vector applied to that function's pullback! (confusing, I know)
> (Pushforward of vector)(function) = Vector(pullback of function).

### Pullbacks of 1-forms

This line of thought can be generalized to covectors, and posteriorly to tensors. Pullback of 1-form, applied to vector = 1-form applied to the pushforward of that vector:

$$(\phi_{t*} \omega)(X) = \omega(\phi_t^* X).$$

> (Pullback of 1-form)(vector) = (1-form)(pushforward of vector).

### Pushforwards of everything

Identifying 

$$\boxed{\phi_{t*} = \phi_{-t}^* }$$

we can define pushforwards for any tensors. For example, let $T^a_b$ be a (1,1)-tensor in abstract index notation. I'll write $T(X,\omega)$ as its explicit action on a one-form and a vector. It follows that

$$(\phi_t^*T)(X,\omega):= T(\phi_{t*}X, \phi_{t*}\omega)=T(\phi_{-t}^*X, \phi_{t*}\omega).$$

All of the quantities in the RHS are well-defined (pushforward on vector, pullback on 1-form). 

## Lie derivatives

A big chunk of what we do below follows [this reference](https://www.math.brown.edu/cdaly2/Notes/Lie_Derivative.pdf).

First, we write $T_p$ for a vector field evaluated at a point $p$: this notation will make explicit at which point where the tensor is defined.

Also, for pushforwards, we write down a subscript showing where the pushforward "acts on". For example: if $\phi_t: p \mapsto \phi_t(p)$, then the pushforward will be written with a subscript $p$ as well:

$$(\phi_t^*)_p: X_p \mapsto (\phi_t^*)_p X \in T_{\phi_t(p)}M.$$

In this notation, we also make explicit the pullback, which "departs" from $\phi_t(p)$:

$$\boxed{(\phi_{-t}^*)_{\phi_t(p)}:T_{\phi_t(p)} M \to T_p M.}$$ 

Then, we define the Lie derivative once and for all as

$$\boxed{(\mathcal L_X T)_p := \lim_{t\to 0}\frac{(\phi_{-t}^*)_{\phi_t(p)}(T_{\phi_t(p)}) - T_p }{t}.}$$ 

This also fits Carroll's formula B.4-B.5; he uses $\phi_{t*}$ instead (also notice that his positions of the asterisk in pullbacks / pushforwards is opposite to ours). We also made explicit the origin of the pushforward: it starts from $\phi_t(p)$. 

Does this definition make sense? The tensor $T_p$ is explicitly defined at point $p$; the other term, $T_{\phi_t(p)}$, is not, but it is "brought back" to $p$ via the pullback operation, which is "based" on $\phi_t(p)$ and drags the tensor back by a parameter value of $-t$, effectively arriving at $p$. All is well-defined.

### Example of applying this formula

Let's do this calculation for a function. *As a tensor*, it does not vary from point to point, even though its value when *calculated* at different points does. What I mean is that $f_p = f_q = f$ for any two points $p, q$, but usually $f(p) \neq f(q)$. As such, using the equation above, we can use the pullback-based expression $\phi_{t*}f = f \circ \phi_t$, hence

$$(\mathcal L_X f)(p) = \lim_{t\to 0}\frac{f(\phi_t(p)) -  f(p)}{t}.$$

To calculate the expression on the RHS, go to a local coordinate chart where $p$ has coordinates $\vec x$ and $\phi_t(p) = \vec x + t \vec X$ where $\vec X$ are the local coordinates of the vector field generating $\phi_t$. Then, Taylor expanding, we get a RHS of $X^i \partial_i f$ which is just $X(f)$ calculated at $p$. It follows that 

$$\mathcal L_X f = X(f).$$

### Calculating the Lie derivative of a vector by hand

Well then; what about acting on vectors? The spoiler is: 

$$\mathcal L_X Y = [X, Y];$$

let's see if this holds. Nothing like a nice manual calculation before we give the more general formula.

Let $M= \mathbf R^2$ and consider a base point $p = (x_0, y_0)$. Let 

$$X = - y \frac{\partial}{\partial x} + x \frac{\partial}{\partial y},\qquad Y=\frac{\partial}{\partial x}.$$

These are two vector fields. At every point of the manifold, they take values - for example, at $p = (x_0, y_0)$, 

$$X_p = - y_0 \left(\frac{\partial}{\partial x}\right)_p + x_0 \left(\frac{\partial}{\partial y}\right)_p \in T_pM,$$

where the subscript $(.)_p$ doesn't mean much in this particular case for the basis vectors (since we are in Euclidean space), but we keep it nonetheless. 

Assume we want to compute $(\mathcal L_X Y)_p$. We need to first solve the equation for the flow of $X$, that is, $\phi_t$; the differential equation for integral curves is 

$$\frac{d}{dt} \binom{x(t)}{y(t)}=\binom{-y(t)}{x(t)}.$$

with solution 

$$x(t) = x_0 \cos t - y_0 \sin t,\quad y(t)=x_0\sin t+ y_0 \cos t.$$

This means that the map $(\phi_t)$ *based on* $p = (x_0, y_0)$ which takes it downstream by a parameter value $t$ is

$$(\phi_t)_{p=(x_0, y_0)} = \binom{x_0 \cos t - y_0 \sin t}{x_0\sin t+ y_0 \cos t}.$$

Notice that in this particular case $\phi_t$ is linear - this won't always be the case. The pushforward is obtained by calculating the Jacobian w.r.t the base coordinates of $p$ (which justifies the often-used notation $d(\phi_t)_p$ for the pushforward). Hence 

$$(\phi_t^*)_p=\begin{pmatrix} \cos t & -\sin t \\ \sin t & \cos t\end{pmatrix},$$

and, for the opposite direction,

$$(\phi_{-t}^*)_{\phi_t(p)}=\begin{pmatrix} \cos t & \sin t \\ -\sin t & \cos t\end{pmatrix}.$$

Notice how the sign has changed with the formal replacement of $t \mapsto -t$. Also, we changed the subscript from $p$ to $\phi_t(p)$ for clarity. 

Note: an important point here is the use of matrices. In the expression for $(\phi_{t})_{p}$, we used matrices as a compact notation of how coordinate systems changed. For the pushforward, however, the matrices actually act on vectors in one space and give vectors in another space. Explicitly, 

$$(\phi_t^*)_p : T_p M \to T_{\phi_t(p)} M.$$

We have all the ingredients we need. We want to calculate

 $$(\mathcal L_X Y)_p := \lim_{t\to 0}\frac{(\phi_{-t}^*)_{\phi_t(p)}(Y_{\phi_t(p)}) - Y_p }{t}.$$

First, we calculate $Y_{\phi_t(p)}$. It is simply 

$$Y_{\phi_t(p)} = \left(\frac{\partial}{\partial x}\right)_{\phi_t(p)}.$$

Then

$$(\phi_{-t}^*)_{\phi_t(p)}(Y_{\phi_t(p)}) = \begin{pmatrix} \cos t & \sin t \\ -\sin t & \cos t\end{pmatrix} \binom{1}{0} = \cos t \left(\frac{\partial}{\partial x}\right)_p-\sin t\left(\frac{\partial}{\partial y}\right)_p.$$

Finally, plugging back into the definition of the Lie derivative, we get

$$(\mathcal L_X Y)_p =\lim_{t\to 0}\frac 1t\left[(\cos t-1)\left(\frac{\partial}{\partial x}\right)_p - \sin t \left(\frac{\partial}{\partial y}\right)_p  \right]=-\left(\frac{\partial}{\partial y}\right)_p$$ 

and we are done. 

The intuition here is as follows: we are trying to calculate the derivative of the vector (field) $Y = \partial/\partial x$ along the integral curves of $X$, which are circles. Notice how there is a difference exactly on the vertical direction between $Y$ and its pullback, which is slightly inclined due to rotation.

By the way: since we already know the spoiler that $\mathcal L_X Y = [X, Y]$, it is worth checking if it works. Recalling that 

$$[X, Y] = \left(X^a\frac{\partial Y^b}{\partial x^a} - Y^a\frac{\partial X^b}{\partial x^a} \right) \frac{\partial}{\partial x^b},$$

we have, for $X^1 = -y, X^2 = x$ and $Y^1 = 1, Y^2 = 0$, that the only non-zero component of the commutator is 

$$[X, Y]^2 = - 1 \frac{\partial x}{\partial x} = -1,$$ 

yielding $[X,Y] = - \partial/\partial y$, as we obtained.

### Proof of the commutator formula

First, notation. The field $Y$ will be written as 

$$Y_p = Y^i(x_p) \left(\frac{\partial}{\partial x^i} \right)_p$$ 

where $x_p$ are the coordinates of point $p$. Let us write the coordinates at $p$ as $(x_0^i)$, generically, and those at $\phi_t(p)$ as $(\tilde x_0^i)$, ie. with a tilde.

The first important thing is to relate the coordinates of $p$ with those of $\phi_t(p)$, in the limit where $t$ is small and can be kept to first order. By definition, 

$$\begin{align*}
\phi_t(p)^i &\equiv \tilde x^i = x_0^i +t \left.\frac{dx^i}{dt}\right|_{t=0}\\
&=x_0^i + tX^i_0(x_0).\qquad (*)
\end{align*}$$

Notice that we explicitly wrote $X_0^i$ as a functon of the coordinates at $p$ - this will come as important later in the calculation of the Jacobian for the pushforward.

With that being done, we follow the algorithm and calculate $Y$ at point $\phi_t(p)$, in the limit where $t$ is small. For this, we use equation $(*)$ above:

$$\begin{align*} 
Y_{\phi_t(p)} &= Y^i(\tilde x) \left(\frac{\partial}{\partial \tilde x^i} \right)_{\phi_t(p)}\\
&=Y^i(x_0+tX_0)\left(\frac{\partial}{\partial \tilde x^i} \right)_{\phi_t(p)}\\
&=\left(Y^i(x_0)+tX_0^j \left.\frac{\partial Y^i(x)}{\partial x^j}\right|_{x=x_0} \right) \left(\frac{\partial}{\partial \tilde x^i} \right)_{\phi_t(p)} + O(t^2)\\
&\equiv(Y^i_0+t X_0^j \partial_jY^i_0) \left(\frac{\partial}{\partial \tilde x^i} \right)_{\phi_t(p)}
\end{align*}$$

where we Taylor-expanded in the third line, and cleaned up notation a bit in the fourth line. 

To calculate the pushforward, we need to calculate the Jacobian of $\phi_t(p)$ with respect to the coordinates $x_0^i$. We can again use equation $(*)$ here; it is easy to see that

$$[(\phi^*_t)_p]^i_j = \delta^i_j+t\, \partial_jX_0^i.$$

Inverting this yields, to first order, 

$$[(\phi^*_{-t})_{\phi_t(p)}]^i_j = \delta^i_j-t\, \partial_jX_0^i.$$

We are ready: the Lie deriative can be calculated as

$$\begin{align*}
\frac{[(\phi^*_{-t})_{\phi_t(p)}]^i_j Y_{\phi_t(p)}^j - Y_p^i}{t} &= \frac{(\delta^i_j-t\, \partial_jX_0^i) (Y^j_0+t X_0^k \partial_kY^j_0)-Y_0^i}{t}\\
&=\frac{Y_0^i+t X_0^k\partial_k Y_0^i-tY_0^j\partial_jX_0^i-Y_0^i+O(t^2)}{t}\\
&=X_0^j\partial_j Y_0^i-Y_0^j\partial_jX^i_0+O(t)\\
&\overset{t\to0}{\longrightarrow} X_0^j\partial_j Y_0^i-Y_0^j\partial_jX^i_0\\
&=[X,Y]^i_p.
\end{align*}$$

Thus, we have the proof! 

> Also check [this reference here](https://cefns.nau.edu/~schulz/lieder.pdf) for a proof without using Taylor series.



