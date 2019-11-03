# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <div style='background-image: url("../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Inverse Problems</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Introduction to Full Wave Inversion </div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Peter Mora [mail](wolop2008@gmail.com)
#
# ---

# ## The method of least-squares
#
# In mathematical texts, least-squares is typically presented in notation as follows.
# Consider a matrix equation
#
# \begin{equation}
# {\bf A} {\bf x} \ = \ {\bf b} \ \ \ ,\ \ \ \tag{1}
# \end{equation}
# where
# \begin{equation}
# {\bf A} \ = \ \left[
# \begin{array}{ccccc}
# A_{11}  & \dots  & A_{1j} & \dots  & A_{1m} \\
# \vdots  & \ddots & \vdots     & \  & \vdots \\
# A_{i1}  & \dots  & A_{ij} & \dots  & A_{im} \\
# \vdots  & \      & \vdots & \ddots & \vdots \\
# A_{n1}  & \dots  & A_{nj} & \dots  & A_{nm} \\
# \end{array}
# \right]
# \ \ \ , \ \ \ 
# {\bf x} \ = \ \left[
# \begin{array}{c}
# x_{1}  \\
# \vdots \\
# x_{j}  \\
# \vdots \\
# x_{m}
# \end{array}
# \right]
# \ \ \ , \ \ \ 
# {\bf b} \ = \ \left[
# \begin{array}{c}
# b_{1}  \\
# \vdots \\
# b_{i}  \\
# \vdots \\
# b_{n}
# \end{array}
# \right] \ \ \ .\ \ \ \tag{2}
# \end{equation}
#
# In this equation, $n$ is the number of equations = number of rows of matrix ${\bf A}$ = number of rows of the column vector $\bf b$, and $m$ is the number of unknowns = elements of the column vector ${\bf x}$.
# Let us assume here there are more equations than unknowns (ie. $n>m$). In Equation (1), the left hand side represents a linear theory that allows us to compute vector ${\bf b}$ which represents a set of observations. Now, assuming that the observations are not exact but have some errors, we can define the sum of the square error as
#
# \begin{equation}
# \varepsilon \ = \ \left( {\bf A} {\bf x} - {\bf b} \right)^T  \left( {\bf A} {\bf x} - {\bf b} \right) \ \ \ .\ \ \ \tag{3}
# \end{equation}
#
# The error is minimized when the derivatve of the error with respect to the unknowns ${\bf x}$ is zero. Namely, when
#
# \begin{equation}
# {{\partial \varepsilon} \over {\partial {\bf x}}} \ = \ 2 {\bf A}^T \left( {\bf A} {\bf x} - {\bf b} \right) \ = \ 
# \left[
# \begin{array}{c}
# 0  \\
# \vdots \\
# 0  \\
# \vdots \\
# 0
# \end{array}
# \right] \ \ \ .\ \ \ \tag{4}
# \end{equation}
#
# The solution is obtained by rearranging Equation (4)
#
# \begin{equation}
# {\bf A}^T {\bf A} {\bf x} \ = \ {\bf A}^T {\bf b} \ \ \ ,\ \ \ \tag{5}
# \end{equation}
#
# which is denoted as the normal equations and hence,
#
# \begin{equation}
# {\bf x} \ = \ \left( {\bf A}^T {\bf A} \right)^{-1} {\bf A}^T {\bf b} \ \ \ .\ \ \ \tag{6}
# \label{eq:ls-soln}
# \end{equation}
#
#
# A solution is possible provided the symmetric Hessian matrix, ${\bf A}^T {\bf A}$, is invertable.
# Hence, the solution that minimizes the square error sum (ie. least-squares solution) can be obtained by solving Equation (6), provided we have a linear theory ${\bf A} {\bf x}$ relating the unknowns ${\bf x}$ to the observables ${\bf b}$.
#
# Clearly, this requires that the matrix ${\bf A}^T {\bf A}$ has an inverse. A typical example of least-squares in mathematics and science is polynomial curve fitting in which case, Equation (1) in subscript notation is given by
#
# \begin{equation}
# \sum_{j=0}^{m-1} x_i^j c_i \ = \ y_i \ \ \ , \ \ \ i = 0,\dots , (n-1) \ \ \ ,\ \ \ \tag{7}
# \end{equation}
#
#
# where the $c_i$ are the coefficients of the polynomial curve which is of order $(m-1)$, and $n$ is the number of
# known $(x_i ,y_i )$ pairs. Hence, the matrix equation is given by
#
# \begin{equation}
# {\bf A} {\bf x} \ = \ \left[
# \begin{array}{ccccc}
# x_0^0     & \dots  & x_0^{j}   & \dots  & x_0^{m-1} \\
# \vdots    & \ddots & \vdots    & \      & \vdots    \\
# x_i^0     & \dots  & x_i^j     & \dots  & x_i^{m-1} \\
# \vdots    & \      & \vdots    & \ddots & \vdots    \\
# x_{n-1}^0 & \dots  & x_{n-1}^j & \dots  & x_{n-1}^{m-1} \\
# \end{array}
# \right]
# \left[
# \begin{array}{c}
# c_0    \\
# \vdots \\
# c_j    \\
# \vdots \\
# c_{m-1}
# \end{array}
# \right]
# \ = \ 
# \left[
# \begin{array}{c}
# y_0    \\
# \vdots \\
# y_j    \\
# \vdots \\
# y_{n-1}
# \end{array}
# \right]
# \ = \ {\bf b} \ \ \ ,\ \ \ \tag{8}
# \end{equation}
#
# in which case, the Hessian matrix is given by
#
# \begin{equation}
# {\bf H} \ = \ {\bf A}^T {\bf A} \ = \ \left[
# \begin{array}{ccccc}
# n              & \dots  & \sum x_i^j       & \dots  & \sum x_i^{m-1}    \\
# \vdots         & \ddots &  \vdots          & \      & \vdots            \\
# \sum x_i^j     & \dots  & \sum x_i^{2j}    & \dots  & \sum x_i^{j+m-1}  \\
# \vdots         & \      & \vdots           & \ddots & \vdots            \\
# \sum x_i^{m-1} & \dots & \sum x_i^{j+m-1} & \dots & \sum x_i^{2(m-1)}
# \end{array}
# \right] \ \ \ ,\ \ \ \tag{9}
# \end{equation}
#
# where the sums are taken from $i=0$ to $i = (n-1)$. 
# The matrix ${\bf H} = {\bf A}^T {\bf A}$ given above is typically well behaved and invertable, but for numerical precision, it is best to rescale $x_i$ such that $x_i \in [-1, 1]$.
# Hence, the polynomial coefficients $c_i$ can be obtained by solving Equation (6).
#
# In geophysical problems, it is generally not possible to express the forward and inverse problems as matrix equations such as Equation (1) for the forward problem, and Equation (6) for the inverse problem. Nonetheless,
# matrix equations provide a useful and concise notation to describe the inverse problem in geophysics, and to
# illustrate how to solve such problems. As such, the examples shown in the following section to demonstrate the characteristics of the different nonlinear least-squares (maximum-liklihood) inversion algorithms will be specified as matrix equations.
# Furthermore, these probems will involve inversion for only two model parameters so that the objective function can be visualized in 2D on the figures, as well as the trajectory of iterations on the objective function.

# ## Probabilistic solution
#
# The inverse problem can be expressed in terms of finding the most probable model ${\bf m}$ for a given set of data observations
# ${\bf d}_0$. Several texts provide a detailed review of inverse methodology (eg. Menke, 2012, Tarantola, 2005. In the following,
# the principles of solving a nonlinear inverse problem are reviewed and examples are provided.
#
# Consider the Gaussian probability density function
# $$
# P \ \propto \ \exp \left[ - {1 \over 2} \left( \Delta {\bf d}^T {\bf C_d^{-1}} \Delta {\bf d} + \Delta {\bf m}^T {\bf C_m^{-1}} \Delta {\bf m} \right) \right] \ = \ \exp \left( - \varepsilon \right) \ \ \ ,
# $$
# where ${\bf C}_{\bf d}$ is the data covariance matrix, ${\bf C}_{\bf m}$ is the model covarience matrix, and
# \begin{equation}
# \Delta {\bf d} = {\bf d} ( {\bf m} ) - {\bf d}_0 \ \ \ , \ \ \ \
# \Delta {\bf m} = {\bf m} - {\bf m}_0 \ \ \ ,
# \end{equation}
# are respectively the data mismatch or residual defined as the difference between the calculated
# data ${\bf d} ( {\bf m})$ and observed data ${\bf d}_0$, and the model mismatch between the
# model ${\bf m}$ and an a priori model denoted ${\bf m}_0$. The most probable or maximum liklihood solution, when the
# probability $P$ is maximal, occurs when the derivative of the square error functional
# \begin{equation}
# \varepsilon \ = \ {1 \over 2} \left( \Delta {\bf d}^T {\bf C_d^{-1}} \Delta {\bf d} + \Delta {\bf m}^T {\bf C_m^{-1}} \Delta {\bf m} \right) \ \ \ ,
# \end{equation}
# is minimized. Hence, we seek a solution when the derivative of the error functional or gradient vector ${\bf g} = {{\partial \varepsilon} / {\partial {\bf m}^T}} = 0$. Namely, we seek the solution to
# \begin{equation}
# {\bf g} \ = \ {{\partial \varepsilon} \over {\partial {\bf m}^T}} \ =  \
# {\bf D}^T {\bf C_d^{-1}} \Delta {\bf d} + {\bf C_m^{-1}} \Delta {\bf m} \ = \ 0 \ \ \ ,\ \ \ \tag{10}
# \end{equation}
# where ${\bf D } = \partial {\bf d} / \partial {\bf m}$ is the Frechet derivative matrix. In many geophysical problems, the number of observations and discrete model parameters defining a region of the Earth is large, and it is too demanding on computer time and
# memory to calculate the components of matrix ${\bf D}$. However, if the operation of ${\bf D}$ and its adjoint ${\bf D}^T$ can
# be calculated, it is possible to solve Equation (10).
# This will be demonstrated in subsequent examples.
#
# Consider the case where it is possible to linearize the theoretical relationship between data and model parameters. In this case, we can substitute $\Delta {\bf d} = {\bf d} ({\bf m}_0 ) + {\bf D} ({\bf m} - {\bf m}_0 ) - {\bf d}_0$ into Equation (10) to obtain
# \begin{equation}
# {\bf D}^T {\bf C}_{\bf d}^{-1} \left[ {\bf d} ( {\bf m}_0 ) + {\bf D} ( {\bf m} - {\bf m}_0  ) - {\bf d}_0 \right] \ + \ {\bf C}_{\bf m}^{-1} ( {\bf m} - {\bf m}_0 ) \ = \ 0 \ \ \ ,
# \end{equation}
# which leads to
# \begin{equation}
# \left( {\bf D}^T {\bf C}_{\bf d}^{-1} {\bf D} + {\bf C}_{\bf m}^{-1} \right)  ( {\bf m} - {\bf m}_0 ) = - {\bf D}^T {\bf C}_{\bf d}^{-1} \left( {\bf d} ( {\bf m}_0 ) - {\bf d}_0 \right) \ \ \ . \ \ \ \tag{11}
# \end{equation}
# The solution to Equation (11) is given by
# \begin{equation}
# {\bf m} \ = \ {\bf m}_0 \ - \ {\bf H}^{-1}   {\bf D}^T {\bf C}_{\bf d}^{-1} \left( {\bf d} ( {\bf m}_0 ) - {\bf d}_0 \right) \ \ \ ,\ \ \ \tag{12}
# \end{equation}
# where the Hessian matrix ${\bf H}$ is given by
# \begin{equation}
# {\bf H} \ = \ \left( {\bf D}^T {\bf C}_{\bf d}^{-1} {\bf D} + {\bf C}_{\bf m}^{-1} \right) \ \ \ .\ \ \ \tag{13}
# \end{equation}
# Equation (12) yields the solution assuming the forward problem is linear such as ${\bf d} = {\bf D} {\bf m}$.
# In this case, and comparing Equation (12) to its equivalent written in the standard notation used in mathematical texts, Equation (6), we see that ${\bf d}_0$ corresponds to the vector ${\bf b}$, ${\bf m}$ corresponds to vector ${\bf x}$, and ${\bf D}$ corresponds to matrix ${\bf A}$. In the standard mathematical notation, the identity matrix ${\bf I}$ replaces the data covariance matrix ${\bf C}_{\bf d}$, and the inverse model covarience matrix ${\bf C}_{\bf m}^{-1}$ is null which means the model vector is unconstrained.
#
# In cases where the Hessian matrix can be calculated,
# setting ${\bf C}_{\bf m}^{-1} = (1/\sigma_m^2 ) {\bf I}$ where $\sigma_m$ is large, will add a small diagonal term to the Hessian matrix thereby stabalizing the matrix inversion without adding any significant constraint to the model parameters.
# This is generally not required for well posed and overdetermined inverse problems, and in any event, it is often not practical in geophysics to calculate the Hessian matrix due to its typically enormous size and consequently, the extreme computational cost to calculate the Hessian.
#
# In practice, the inverse data covariance matrix ${\bf C}_{\bf d}^{-1}$ is usually replaced by a constant diagonal matrix $(1/\sigma_d^2 ){\bf I}$ and the model inverse covariance matrix is set to null, ie. ${\bf C}_{\bf m}^{-1} = (1/\sigma_m^2 ) {\bf I} = 0 \Rightarrow \sigma_m = \infty$. Hence, there is no constraint on the model parameters which are resolved solely by the data. A non-constant diagonal matrix ${\bf C}_{\bf d}$ can be used to allow for different noise levels for the different data observations. For example, in the case of data having two noise levels with variance $\sigma_1$ and $\sigma_2$, then a data covariance matrix of form
#
# \begin{equation}
# {\bf C}_{\bf d} \ = \ \left[
# \begin{array}{cc}
# \sigma_1^2 {\bf I} & 0 \\
# 0                  & \sigma_2^2 {\bf I}
# \end{array}
# \right] \ \ \ \Rightarrow \ \ \ 
# {\bf C}_{\bf d}^{-1} \ = \ \left[
# \begin{array}{cc}
# {\bf I} / \sigma_1^2 & 0 \\
# 0                  & {\bf I } / \sigma_2^2
# \end{array}
# \right] \ \ \ ,
# \end{equation}
#
# can be specified to allow for these differing levels of noise. This corresponds to "weighted least-squares", with the meaning of
# the two different weights corresponding to the two different noise variances in the data.

#
#
# ### Newton method
#
# In the general case, when the forward problem is nonlinear, but when the Hessian can be calculated, then the least-squares = maximum liklihood solution can be obtained by iteratively updating a model estimate.
# For example, Equations (12) and (13) can be solved by iteratively updating the model ${\bf m}$, starting from an initial estimate ${\bf m}_0$ until the solution converges. Let ${\bf m}_n$ be the model parameters at iteration $n$. Replacing ${\bf m}$ with ${\bf m}_{n+1}$, and ${\bf m}_0$ with ${\bf m}_n$, Equation (12) becomes
#
# \begin{equation}
# {\bf m}_{n+1} \ = \ {\bf m}_n \ - \ {\bf H}_n^{-1}   {\bf g}_n \ \ \ ,\ \ \ \tag{14}
# \end{equation}
#
# where ${\bf g}_n$ is the gradient vector at iteration $n$ given by
#
# \begin{equation}
# {\bf g}_n \ = \ {\bf D}_n^T {\bf C}_{\bf d}^{-1} \Delta {\bf d}_n \ \ \ ,\ \ \ \tag{15}
# \end{equation}
#
# and ${\bf H}_n$ is the Hessian matrix given by Equation (13) at the $n$-th iteration, and
# the data residual at the $n$-th iteration is given by
#
# \begin{equation}
# \Delta {\bf d}_n \ = \ {\bf d} ( {\bf m}_n ) - {\bf d}_0 \ = \ {\bf d}_n - {\bf d}_0 \ \ \ .
# \end{equation}
#
# Equation (14) specifies a Newton or Gauss-Newton method for nonlinear least-squares.
# Note that the definition of the gradient vector Equation (15) is identical to the previous definition for the gradient vector given in Equation (10) for the case with ${\bf C}_{\bf m}^{-1} = 0$, ie. unconstrained model parameters ($\sigma_m = \infty$).
#
# The Newton method converges in one iteration for linear problems, and rapidly for nonlinear problems provided the starting model is sufficiently close to the solution. The Newton method approximates the objective function
# with the parabolic form at each iteration and hence, it requires the Hessian matrix to be calculated and inverted at each iteration. 
# Unfortunately, it is rarely possibly in most real geophysics problems to be able to calculate the Hessian matrix ${\bf H}$ or its inverse ${\bf H}^{-1}$. For this reason, other methodologies are required which do not require the Hessian matrix. Sometimes it is possible to derive a preconditioner to the gradient that approximates the effect of multiplication by the inverse Hessian matrix in which case, the rate of convergence can be improved and in some cases, approach the effect of the inverse Hessian. When it is possible to derive a preconditioner that approximates the effect of the inverse Hessian, Equation (14) becomes
#
# \begin{equation}
# {\bf m}_{n+1} \ = \ {\bf m}_n \ - \ {\bf P}_n   {\bf g}_n \ \ \ ,
# \end{equation}
#
# where the preconditioner denoted ${\bf P}$ approximates the inverse Hessian, ie. ${\bf P} \sim {\bf H}^{-1}$.

# ### Steepest descent method
#
# In cases where the Hessian matrix ${\bf H}$ cannot be calculated as is the case for many geophysical inverse problems, one can use the method of steepest descent. In this case, the model parameters are iteratively updated in the direction of steepest descent of the objective function = negative of the gradient vector. Namely, we solve
#
# \begin{equation}
# {\bf m}_{n+1} \ = \ {\bf m}_n \ - \ \alpha_n {\bf g}_n \ \ \ ,\ \ \ \tag{16}
# \end{equation}
#
# where the scalar $\alpha_n$ is the steplength and must be solved by a line search to seek the minimum of the objective function in direction ${\bf g}$. An estimate of $\alpha$ can be made if one assumes linearity in
# the direction of steepest descent ${\bf -g}$. Specifically, setting the data calculation for the next iteration to be
#
# \begin{equation}
# {\bf d} ({\bf m}_{n+1}) \ = \ {\bf d} ({\bf m}_n ) \ - \ \alpha_n {\bf D}_n {\bf g}_n \ \ \ ,
# \end{equation}
#
# and solving for when the derivative of the objective function with respect to scalar $\alpha$ is zero. The sum of square error is given by
#
# \begin{equation}
# \varepsilon \ = \ \left( {\bf d} - \alpha {\bf D} {\bf g} - {\bf d}_0 \right)^T {\bf C}_{\bf d}^{-1} \left( {\bf d} - \alpha {\bf D} {\bf g} - {\bf d}_0 \right) \ + \ \left( {\bf m} - \alpha {\bf g} - {\bf m}_0 \right)^T \left( {\bf m} - \alpha {\bf g} - {\bf m}_0 \right) \ \ \ .
# \end{equation}
#
# The optimal steplength can be calculated by solving ${{\partial \varepsilon} / {\partial \alpha}} = 0$ which leads to the result
#
# \begin{equation}
# \alpha \ = \ { {{\bf g}^T {\bf g} } \over { {\bf g}^T {\bf D}^T {\bf C}_{\bf d}^{-1} {\bf D} {\bf g} + {\bf g}^T {\bf C}_{\bf m}^{-1} {\bf g} }} \ \ \ ,\ \ \ \tag{17}
# \end{equation}
#
# where in Equation (17), the gradient vector ${\bf g}$ is as specified by Equation (10) and both the gradient vector ${\bf g}$ and linearization ${\bf D} = {\partial {\bf d} / \partial {\bf m}}$ are calculated at the $n$-th iteration about model ${\bf m} = {\bf m}_n$, ie. ${\bf g} = {\bf g}_n$ and ${\bf D} = {\bf D}_n$.
#
# The method of steepest descent allows a solution to be obtained without the need to calculate the Hessian matrix and its inverse.
# However, it can be slow to converge when the model parameters are not equally well resolved and there is nonlinearity. In this case, the objective function will be banana shaped and the steepest descent iterations may follow a slow zig-zag path down the valley of the banana shaped valley to the minimum of the objective function. For this reason, a modification of the steepest descent method should be made. The conjugate gradient method modifies the steepest descent direction as a linear combination of the past and present gradient directions, and avoids the zig-zag problem often encountered by the steepest descent method.

# ### Conjugate gradient method
#
# The conjugate gradient method starts by an iteration in the direction of steepest descent. Subsequently, iterations are in a conjugate direction which is a linear combination of the current gradient vector and the last conjugate direction, namely,
#
# \begin{equation}
# {\bf c}_n \ = \ {\bf g}_n + \beta_n {\bf c}_{n-1} \ \ \ ,
# \end{equation}
#
# where ${\bf c}_n$ is used to denote the conjugate direction at the $n$-th iteration, and $\beta_n$ is a scalar that can be calculated according to a number of different formulas which will be specified in the following. The benefit of the conjugate gradient method is that avoids the zig-zag behaviour that the steepest descent suffers from.
#
# The value of the scalar $\beta_n$ can be calculated by different formulas which are named after their developers.
# Four polular choices are the Fletcher-Reeves, Polak-Ribiere, Hestenes-Steifl, and Dau-Yuan methods (denoted as FR, PR, HS and DY). Each of these formulas yield the same result for quadratic objective functions (linear theories), and result in the solution after exactly $n$ iterations where $n$ is the number of unknowns or model parameters being solved for. 
# However, the performance of these different methods to calculate $\beta_n$ varies. I have tested each of these popular choices for the example of nonlinear inversion shown in the next section of this chapter, and found that for this nonlinear theory relating model parameters to data, the performance (in order of descending performance) is PR, FR, DY and lastly, HS.
# The PR and FR methods were similar in performance with PR being slightly better, and these were both significantly better in performance than the DY and SH methods which were also similar in performance.
# For this reason, only the formulas for the Polak-Ribiere and Fletcher-Reeves methods are provided in this chapter and are given by 
#
# \begin{equation}
# \beta_n^{PR} \ = \ { {{\bf g}_n^T ( {\bf g}_n - {\bf g}_{n-1} ) } \over {\bf g}_{n-1}^T {\bf g}_{n-1}}
# \ \ \ , \ \ \ 
# \beta_n^{FR} \ = \ { {{\bf g}_n^T {\bf g}_n } \over {\bf g}_{n-1}^T {\bf g}_{n-1}} \ \ \ .\ \ \ \tag{18}
# \end{equation}
#
# At some stage progress using the conjugate gradient method may slow or stop due to loss of conjugacy for
# nonlinear functions (ie. non-quadratic objective functions).
# If this occurs, a reset to the direction of steepest descent will allow subsequent iterations to regain conjugacy. A popular choice for $\beta$ that provides an automatic direction reset and, provides the best performance for the example in this chapter, is given by
#
# \begin{equation}
# \beta \ = \ \max \left\{ 0,\beta^{PR} \right\} \ \ \ .\ \ \ \tag{19}
# \end{equation}
#
# The conjugate gradient method involves iterations updating the model in the conjugate gradient direction {\bf c} as follows
#
# \begin{equation}
# {\bf m}_{n+1} \ = \ {\bf m}_n \ - \ \alpha_n {\bf c}_n \ \ \ ,\ \ \ \tag{20}
# \label{eq:conjugate-gradient-iterations}
# \end{equation}
#
# where the optimal steplength can be calculated by a line search, with an initial estimate of $\alpha_n$ using Equation (17) with the gradient ${\bf g}$ being replaced by the conjugate gradient ${\bf c}$. Namely,
#
# \begin{equation}
# \alpha \ = \ { {{\bf c}^T {\bf c} } \over { {\bf c}^T {\bf D}^T {\bf C}_{\bf d}^{-1} {\bf D} {\bf c} + {\bf c}^T {\bf C}_{\bf m}^{-1} {\bf c} }} \ \ \ .
# \end{equation}
#
# If a preconditioner that approximates the inverse Hessian can be derived, Equation (20) becomes
#
# \begin{equation}
# {\bf m}_{n+1} \ = \ {\bf m}_n \ - \ \alpha_n {\bf p}_n \ \ \ ,\ \ \ \tag{21}
# \end{equation}
# where ${\bf p} = {\bf P} {\bf c}$ is the preconditioned conjugate gradient direction, with ${\bf P} \approx {\bf H}^{-1}$ being
# the preconditioner.

# ## A simple example problem
#
# In the case of overdetermined inverse problems where the number of data observations
# exceeds the number of model parameters, nonlinear inversion of geophysical data can be achieved by
# iterative algorithms such as the Newton method (provided the Hessian and its inverse can be calculated) given by
# Equation (14), the steepest descent method given by Equation (16), or the conjugate
# gradient method given by Equation (20). In the following, we solve the same problem with each of
# these three methods to show their performance for a simple example with two model parameters.
#
# Consider the following formula to calculate the traveltime of seismic waves down to an interface and back to the Earth's surface for
# the case the a single homogeneous layer of depth $z$ over a halfspace. Consider the case where a seismic source is located at a
# position $x=0$ and that receivers are located along the Earth's surface at positions $x$.
# The formula for the traveltime down to the interface at depth $z$ and back to a receivers on the Earth's surface
# at an offset $x$ from the source is given by Pythagoras's theorem. Hence, the forward problem to calculate traveltimes is
# \begin{equation}
# t^2 ( x,v) \ = \ \left( {z \over v} \right)^2 \ + \ \left( {x \over v} \right)^2 \ \ \ .\ \ \ \tag{22} 
# \end{equation}
#
# ## Linear solution and linearization
#
# Note that the forward problem specified by Equation (22) is nonlinear in terms of the model parameters $z$ and $v$.
# However, in this example, the forward problem can be recast in terms of calculating data defined as
# squared traveltime from model parameters defined as $m_1 = t_0^2$ and $m_2 = 1/v^2$.
# With this choice of model parameters and data variables, the forward problem is linear. Namely, the new forward problem in
# terms of variables $m_1$ and $m_2$ is given by
# \begin{equation}
# d({\bf m}) \ = \ m_1 \ + \ x^2 m_2 \ \ \ ,\ \ \ \tag{23}
# \label{eq:forward-linear}
# \end{equation}
# and leads to a quadratic objective function as a function of the model parameters $m_1$ and $m_2$. Hence, the
# linear forward problem specified by Equation (23) can solved using the Newton method given by Equation (14).
# As the problem is linear, the solution using the Newton method will be obtained in a single iteration.
#
# The linear forward problem written as a matrix equation is given by
#
# \begin{equation}
# {\bf d} \ = \ 
# \left[
# \begin{array}{c}
# \vdots  \\
# d(x_i ) \\
# \vdots
# \end{array}
# \right]
# \ = \ 
# \left[
# \begin{array}{cc}
# \vdots & \vdots \\
# 1 & x_i^2 \\
# \vdots & \vdots
# \end{array}
# \right]
# \left[
# \begin{array}{c}
# m_1 \\
# m_2
# \end{array}
# \right]
# \ = \ 
# {\bf D} {\bf m} \ \ \ .\ \ \ \tag{24}
# \end{equation}
#
# Hence, the solution is given by Equation (14). Namely
#
# \begin{equation}
# {\bf m} \ = \ \left( {\bf D}^T {\bf C}_{\bf d}^{-1} {\bf D} \right) {\bf D}^T {\bf C}_{\bf d}^{-1} {\bf d}_0 \ \ \ ,
# \end{equation}
#
# which solves for the values of $m_1 = t_0^2$ and $m_2 = 1/v^2$. The solution for $m_2$ is then used to calculate $v$ (ie. $v = 1/\sqrt{m_2}$).
# Subsequently, now that the velocity $v$ is known, the solution for $m_1$ can be used to calculate the depth to the interface, namely $z = t_0 v$ where $t_0 = \sqrt{m_1}$. This example is provided to illustrate that in some cases, the original nonlinear inverse problem can be recast as a linear inverse problem through an appropriate change of variables.
# The plots show the objective function for this linear example with the path of the model as a function of iteration shown in yellow. In this example, the forward problem was calculated using the linear equation given by Equation (24), and the inverse problem was calculated by the steepest-descent method given by Equation (16) (top plot), and by the conjugate gradient method using Equation (20) (middle plot), and by the Newton method given by Equation (14) (bottom plot).
#
# One observes that the Newton method results in a solution after one iteration whereas the steepest descent method requires four iterations due to following a  zig-zag path around the elongate valley in the objective function. In contrast, the conjugate gradient method shown in the middle plot requires exactly two iterations = same number of iterations as model parameters.
# Hence, the conjugate gradient method is superior to the steepest descent method, and is always preferable to steepest descent. If an approximate inverse Hessian preconditioner can be derived, the conjugate gradient method
# can be further optimized by replacing the conjugate gradient direction by a preconditined conjugate gradient direction, ${\bf p} = {\bf P} {\bf c}$, in which case, the preconditioned conjugate gradient
# method given by Equation (21) is used in place of Equation (20).

# + {"code_folding": [0]}
# Header & Import Libraries (PLEASE RUN THIS CODE FIRST!) 
#=======================================================================
# Example of a linear least squares problem
#
# Generate colour plots of the objective function and the
# path for the Newton method, steepest descent method,
# and the conjugate-gradient method.
#
# Plot the error and model parameters versus iteration
# for each case, and plot the true data, noisy data,
# and the data solution
#
# Author: Peter Mora
#
# Date:   1/1/2017
#
#=======================================================================

# Import libraries that are needed

import matplotlib
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.random import randn
import random
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D
import warnings
warnings.filterwarnings('ignore')


# + {"code_folding": [0]}
# Functions definition
# -------------------------------------
# define data theory
#
# d = d(x,m) = D m
#
# where D = (1, x^2) and m = (m_1, m_2)^T
#
# so d(x,m) = t0^2 + x^2 / v^2 = m_1 + x^2 *m_2
#
# with m_1 = t0^2 and m_2 = 1/v^2
#--------------------------------------

def d_calc(m,x):
    m1 = m[0]
    m2 = m[1]
    d_calc = m[0] + x**2 * m[1]
    return d_calc


# + {"code_folding": [0]}
# Initializations 
#--------------------------------------
# Main program
#--------------------------------------

# Define the number of data points,
# data spacing dx, and model parameters

nx      = 50
nd      = nx
dx      = 1.0/nx
n_iter = 10        # Number of iterations

t0_soln = 1.1
v_soln  = 0.9
v2_soln = v_soln**2

m       = np.zeros(2)
m_true  = np.zeros(2)
m_true[0] = t0_soln**2
m_true[1] = 1/(v_soln**2)

# Define the spatial grid x and data space d

x       = np.arange(0, nx, dx)
d       = np.zeros(nx)
d_true  = np.zeros(nx)
d_0     = np.zeros(nx)
d_soln  = np.zeros(nx)

# Define the model space and the data
# matrix describing the linear theory

nm    = 2
m     = np.zeros(nm)
im    = np.arange(0, nm, 1)
id    = np.arange(0, nd, 1)
D     = (nd,2)
D     = np.zeros(D)

# Define the Gaussian noise

sigma_d = 0.05
noise = sigma_d * randn(nx, 1)

# Make a zone of high noise

for kx in range(int(nx/3),int(2*nx/3)):
    noise[kx] = 4 * noise[kx]

# Initialize the data and add noise

for kx in range(0,nx):
    d_true[kx] = d_calc(m_true,x[kx])
    d_0[kx]    = d_true[kx] + noise[kx]

# Initialize the linear theory data matrix

for kx in range(0,nx):
    D[kx][0] = 1
    D[kx][1] = x[kx]**2

# Initialize the diagonal data inverse covariance matrix

Cdi = (nd,nd)
Cdi = np.zeros(Cdi)
for id in range(0,nd):
    Cdi[id][id] = 1/noise[id]**2

# Initalize the data matrix transpose and inverse
# Hessian matrix H^-1 = (DT Cd^-1 D)^-1

DT      = D.transpose()
Cdid    = np.dot(Cdi,d_0)
DTd     = np.dot(DT,Cdid)
CdiD    = np.dot(Cdi,D)
DTDi    = inv(np.dot(DT,CdiD))



# + {"code_folding": [0]}
# Computation of the problem
# Calculate the solution by the Newton method

m_soln  = np.dot(DTDi,DTd)
for kx in range(0,nx):
    d_soln[kx] = d_calc(m_soln,x[kx])

# Design model variables at mesh points

t02_min = 0.4
t02_max = 1.4
vi2_min = 0.4
vi2_max = 1.4
i1 = np.arange(t02_min, t02_max, 0.01)
i2 = np.arange(vi2_min, vi2_max, 0.01)
t02, vi2 = np.meshgrid(i1, i2)
nt2 = t02.shape[0]
nv2 = vi2.shape[1]
E   = np.zeros(t02.shape)

# Set the starting model

m_start = [0.75,0.45]                   # Starting model

# Calculate the error functional
# over the model space

for i in range(t02.shape[0]):
    for j in range(t02.shape[1]):
        for k in range(0,nx):
            x2 = x[k]**2
            m[0] = t02[i][j]
            m[1] = vi2[i][j]
            E[i][j] = E[i][j] + (d_calc(m,x[k])-d_0[k])**2

# Path to the solution by the Newton method

t02_i = np.array([m_start[0],m_soln[0]])
v02_i = np.array([m_start[1],m_soln[1]])

# Add the plot of the path to the minimum via
# the steepest descent method labeled _sd

m_sd = np.zeros((n_iter+1,2))            # Model parameters versus iteration
E_sd = np.zeros(n_iter)                  # Error versus iteration

m_sd[0] = m_start                        # Starting model
d = np.zeros(nx)

# Steepest descent iterations

for iter in range(0,n_iter):
    d = np.dot(D,m_sd[iter])
    Delta_d = d - d_0
    E_sd[iter] = np.dot(Delta_d.transpose(),Delta_d)
    g = np.dot(DT,Delta_d)
    Dg = np.dot(D,g)
    gDTDg = np.dot(Dg.transpose(),Dg)
    gTg   = np.dot(g.transpose(),g)
    alpha = gTg/gDTDg
    m_sd[iter+1] = m_sd[iter] - alpha * g

# Calculate the path to the minimum via
# the conjugate gradient method labeled _cg

m_cg = np.zeros((n_iter+1,2))
m_cg[0] = m_start
E_cg = np.zeros(n_iter)

# Conjugate gradient iterations

for iter in range(0,n_iter):

#   Calculate D = linearization about the current model and the data

    d = np.dot(D,m_cg[iter])
    Delta_d = d - d_0

#   Calculate the error and gradient (steepest descent) direction

    E_cg[iter] = np.dot(Delta_d.transpose(),Delta_d)
    g_last = g
    g = np.dot(DT,Delta_d)

#   Compute the new conjugate direction

    if iter == 0:
        g_last = g
        c_last = 0
        beta = 0
    else:
#       Polak-Ribiere is best in examples tried
        beta_PR = np.dot(g.transpose(),(g-g_last))/np.dot(g_last.transpose(),g_last)
#       Fletcher-Reeves is secont best
        beta_FR = np.dot(g.transpose(),g)/np.dot(g_last.transpose(),g_last)
#       Hesyenes-Stiefls (not as good as FR or PR)
        beta_HS = np.dot(g.transpose(),(g-g_last))/np.dot(c_last.transpose(),(g-g_last))
#       Dai-Yuan (better than HS, worse than FR or PR)
        beta_DY = np.dot(g.transpose(),g)/np.dot(c_last.transpose(),(g-g_last))

#       Use the Polak-Ribiere method for beta

        beta = beta_PR

    if beta < 0:
        beta = 0
    c = g + beta * c_last
    c_last = c

#   Estimate of steplength alpha

    Dc = np.dot(D,c)
    cDTDc = np.dot(Dc.transpose(),Dc)
    cTc   = np.dot(c.transpose(),c)
    alpha = cTc/cDTDc

    m_cg[iter+1] = m_cg[iter] - alpha * c



# + {"code_folding": [0]}
# Plots 
#----------------------------------------------------------------
# Plot the true data and calculated
# data at the two solutions: steepest
# descent and conjugate gradient
#----------------------------------------------------------------

fig = plt.figure(1,figsize=(15,10))
ax3 = fig.add_subplot(221)
plt.title('Data plots')
ax3.plot(d_true,'g.-')
ax3.plot(d_0 ,'r--')
ax3.plot(d_soln,'b-')
ax3.legend(['d_true','d_observed','d_solution'], borderaxespad=0.) # bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=3, , mode="expand"
plt.xlabel('x', fontsize=14)     # Label x axis
plt.ylabel('d(x)', fontsize=14)  # Label y axis
#plt.savefig('ls_linear_data.png')
#plt.show()
#----------------------------------------------------------------
# Plot the objective function and path to a solution
# as a colour image for the various methods
#----------------------------------------------------------------

# Allow for the reverseal of y axis in image plots

nv = nt2
ER = np.zeros(t02.shape)
for i in range(t02.shape[0]):
    for j in range(t02.shape[1]):
       ER[i][j] = E[nv-i-1][j]

# Steepest descent method
#plt.figure(2)
plt.subplot(222)
image = plt.imshow(ER, vmax=5, extent=[t02_min, t02_max, vi2_min, vi2_max], cmap='rainbow')
plt.plot(m_sd[:,0],m_sd[:,1],'.-y', ms=8, markeredgewidth=0, lw=2, color='yellow')
plt.colorbar(image)                        # Make colorbar
plt.title('Objective function: steepest descent method')
plt.xlabel('$t_0^2$', fontsize=14)            # Label x axis
plt.ylabel('$1/v^2$', fontsize=14)           # Label y axis
#plt.savefig('ls_linear_objective_sd.png')
#plt.show()

# Conjugate-gradient method
#plt.figure(3)
plt.subplot(223)
image = plt.imshow(ER, vmax=5, extent=[t02_min, t02_max, vi2_min, vi2_max], cmap='rainbow')
plt.plot(m_cg[:,0],m_cg[:,1],'.-y', ms=8, markeredgewidth=0, lw=2, color='yellow')
plt.colorbar(image)                        # Make colorbar
plt.title('Objective function: conjugate gradient method')
plt.xlabel('$t_0^2$', fontsize=14)            # Label x axis
plt.ylabel('$1/v^2$', fontsize=14)           # Label y axis
#plt.savefig('ls_linear_objective_cg.png')
#plt.show()

# Newton method
#plt.figure(4)
plt.subplot(224)
image = plt.imshow(ER, vmax=5, extent=[t02_min, t02_max, vi2_min, vi2_max], cmap='rainbow')
plt.plot(t02_i,v02_i,'.-y', ms=8, markeredgewidth=0, lw=2, color='yellow')
plt.colorbar(image)                        # Make colorbar
plt.title('Objective function: Newton method')
plt.xlabel('$t_0^2$', fontsize=14)            # Label x axis
plt.ylabel('$1/v^2$', fontsize=14)           # Label y axis
#plt.savefig('ls_linear_objective_newton.png')

plt.subplots_adjust(hspace=0.3)
plt.show()
# -

# ## The nonlinear problem
#
# In the following, we do not recast the example as a linear problem, but solve for the depth $z$ and velocity $v$, with data being $t^2$. Hence,
# the forward problem is given by Equation (22). Taking the partial derivative of this forward problem with respect to the two model
# parameters $m_1 = z$ and $m_2 = v$ results in
# \begin{equation}
# {{\partial d } \over {\partial z}} \ = \ {{\partial t^2} \over {\partial z}} \ = \  {{2 z} \over v^2} \ \ \ , \ \ \ 
# {{\partial d } \over {\partial v}} \ = \ {{\partial t^2} \over {\partial v}} \ = \  {{-2 (z^2 + x^2)} \over v^3} \ \ \ ,
# \end{equation}
# Hence, the forward problem written as a matrix equation is given by
#
# \begin{equation}
# {\bf d} \ = \ 
# \left[
# \begin{array}{c}
# \vdots \\
# d ( x_i )   \\
# \vdots
# \end{array}
# \right]
# \ = \ 
# \left[
# \begin{array}{c}
# \vdots \\
# d_i   \\
# \vdots
# \end{array}
# \right]
# \ = \ 
# \left[
# \begin{array}{cc}
# \vdots & \vdots \\
# {{2z} \over v^2} & {{-2(z^2+x_i^2)}\over v^3}\\
# \vdots & \vdots
# \end{array}
# \right]
# \left[
# \begin{array}{c}
# m_1 \\
# m_2
# \end{array}
# \right] \ = \ {\bf D} {\bf m}
# \ \ \ .\ \ \ \tag{25}
# \end{equation}
#
# where the $x_i$ are the locations along the Earth's surface where the traveltime is recorded.
# This equation is used to define the forward problem in the following examples to study
# the behaviours of the Newton, steepest-descent and conjugate gradient methods for nonlinear
# inverse problems.
#
# ### Newton method
#
# In a problem in which the Hessian matrix can be calculated, the Newton method can be used to solve a nonlinear inverse problem.
# In this section, data computed using the forward problem defined by Equation (22) is inverted using the Newton method
# defined by Equation (14) to solve for the model parameters, $z$ and $v$.
# The python code generates data according to
# Equation (22), and the Frechet derivative matrix ${\bf D}$ according to Equation (25).
# The initialization of noisey data is made, and the corresponding
# data covariance matrix is calculated. The top plot shows the noisey data that will be inverted, as well as the data without noise, and the data calculated with the solution. The bottom plot shows the objective function and the path of the iterations
# as the Newton method converges. This demonstrates that the nonlinear inverse problem is solved after about 4 iterations by the
# Newton method.
#
# ### Steepest descent method
#
# The second plot shows the result of applying Equations (16)
# to solve the nonlinear inverse problem by the method of steepest descent. In this example, the size of the steplength, $\alpha$, is
# calculated by Equation (17). One observes that the trajectory of the iterations towards
# the solution for $z$ and $v$ follows a zig-zag path on the objective function (second plot).
# Convergence is slow due to this zig-zag pattern along the banana shaped valley of the objective function. Namely, at least 20-30 iterations are required to approach the solution for $z$ and $v$. For this reason, the steepest descent method is not
# an efficient method to solve inverse problems, especially when these are nonlinear problems as is the usual case in geophysics.
#
# ### Conjugate gradient method
#
# In the conjugate-gradient method, a line search is used to fine the optimal steplength $\alpha$.
# The third plot shows the objective function, and the path of the conjugate gradient iterations
# towards the solution. There is no zig-zag pattern like on the steepest gradient convergence path, and the
# solution is obtained after about 6 iterations, just slightly more than the number of iterations required by the Newton method.
# This plot illustrates the ability of the conjugate gradient method to follow a much more optimal path towards the solution than
# is possible by the steepest descent method, sometimes approaching the efficiency of the Newton method.
#
# These plots demonstrate that the Newton method converges the most rapidly in around 4-5 iterations, and the conjugate gradient method is slightly slower requiring between 5 and 10 iterations. The steepest descent method is the slowest to converge, and is still converging towards the solution after 20 iterations.

# + {"code_folding": []}
# Import Libraries (PLEASE RUN THIS CODE FIRST!) 
import matplotlib
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.random import randn
import random
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D
from IPython.display import HTML, display


# + {"code_folding": [0]}
# Funcitons definition

#--------------------------------------
# define data theory
#--------------------------------------

def d_(m):
    d = np.zeros(nd)
    for k in range(0,nx):
        d[k] = (m[0]/m[1])**2 + (x[k]/m[1])**2
    return d

# Define the linearized data theory: d ~ Dm

def D_(m):
    D = np.zeros((nd,nm))
    for k in range(0,nx):
        D[k][0] =  2*m[0]/m[1]**2
        D[k][1] = -2*(m[0]**2+x[k]**2)/m[1]**3
    return D


# + {"code_folding": [0]}
# Initialization parameters
# Define the number of data points,
# data spacing dx, and model parameters

nx      = 51
nd      = nx
nm      = 2
n_iter  = 20  # number of iterations

z_soln  = 1.1
v_soln  = 0.9

dx      = 1.0/nx
x       = np.arange(0, nx, dx)

m       = np.zeros(2)
m_true  = [z_soln,v_soln]
m_start = [0.25,0.4]

sigma_d = 0.1
noise   = np.zeros(nd)
for k in range (0,nd):       noise[k] = sigma_d * randn(1,1)
for k in range(int(nd/3),int(2*nd/3)): noise[k] = 2 * noise[k]

# Define the Gaussian noise and noisy data denoted d_0

d       = np.zeros(nd)
d_true  = d_(m_true)
d_0     = d_true + noise

D = D_(m_start)

# Data inverse covariance matrix definition

Cdi = np.zeros((nd,nd))
for id in range(0,nd):
    Cdi[id][id] = 1/noise[id]**2

# Model variables at mesh points of objective function

z_min = 0.0
z_max = 1.4
v_min = 0.2
v_max = 1.4
iz = np.arange(z_min, z_max, 0.01)
iv = np.arange(v_min, v_max, 0.01)
z, v = np.meshgrid(iz, iv)

# Define the Gaussian noise and noisy data denoted d_0

d       = np.zeros(nd)
d_true  = d_(m_true)
E0_obj = np.zeros(z.shape)
E1_obj = np.zeros(z.shape)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        m[0] = z[i][j]
        m[1] = v[i][j]
        d = d_(m)
        for k in range(0,nx):
            E0_obj[i][j] = E0_obj[i][j] + (d[k] - d_0[k])**2

# Invert the y axis for imshow

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        E1_obj[i][j] = E0_obj[z.shape[0]-i-1][j]

m_cg = np.zeros((n_iter+1,2))
m_sd = np.zeros((n_iter+1,2))
m_h = np.zeros((n_iter+1,2))
E_cg = np.zeros(n_iter)
E_sd = np.zeros(n_iter)
E_h = np.zeros(n_iter)
m_cg[0] = m_start
m_sd[0] = m_start
m_h[0] = m_start
d_i = np.zeros(nx)
g = np.zeros(nm)
g_last = np.zeros(nm)
c_last = np.zeros(nm)

alpha_min = 0
alpha_max = 4
d_alpha = 0.01
alpha_i = np.arange(alpha_min, alpha_max, d_alpha)
n_alpha = alpha_i.shape[0]
E_cg_i = np.zeros(n_alpha)
m_cg_i = np.zeros((n_alpha,nm))
d_c = np.zeros(nx)

# + {"code_folding": [0]}
# Computing simulation
#-----------------------------------------------------
# Steepest descent iterations
#-----------------------------------------------------

for iter in range(0,n_iter):
    D           = D_(m_sd[iter])
    d           = d_(m_sd[iter])
    DT          = D.transpose()
    Delta_d     = d - d_0
    E_sd[iter]   = np.dot(Delta_d.transpose(),Delta_d)
    g           = np.dot(DT,Delta_d)
    Dg          = np.dot(D,g)
    gDTDg       = np.dot(Dg.transpose(),Dg)
    gTg         = np.dot(g.transpose(),g)

#   Calculate the steplength alpha by the linear estimation method

    alpha       = gTg/gDTDg
    m_sd[iter+1] = m_sd[iter] - alpha * g

#-----------------------------------------------------
# Conjugate gradient iterations
#-----------------------------------------------------

for iter in range(0,n_iter):

#   Calculate D = linearization about the current model and the data

    D       = D_(m_cg[iter])
    d       = d_(m_cg[iter])
    DT      = D.transpose()
    Delta_d = d - d_0

#   Calculate the error and gradient (steepest descent) direction

    E_cg[iter] = np.dot(Delta_d.transpose(),Delta_d)
    g_last    = g
    g         = np.dot(DT,Delta_d)

#   Compute the new conjugate direction

    if iter == 0:
        g_last = g
        c_last = 0
        beta   = 0
    else:

#       Calculate beta for different methods

#       Polak-Ribiere
        beta_PR = np.dot(g.transpose(),(g-g_last))/np.dot(g_last.transpose(),g_last)
#       Fletcher-Reeves (less good than PR)
        beta_FR = np.dot(g.transpose(),g)/np.dot(g_last.transpose(),g_last)
#       Hesyenes-Stiefls (not as good as FR or PR)
        beta_HS = np.dot(g.transpose(),(g-g_last))/np.dot(c_last.transpose(),(g-g_last))
#       Dai-Yuan (better than HS, worse than FR or PR)
        beta_DY = np.dot(g.transpose(),g)/np.dot(c_last.transpose(),(g-g_last))

#       Use the Polak-Ribiere method (best)

        beta = beta_PR
    if beta < 0:
        beta = 0
    c = g + beta * c_last
    c_last = c

#   First estimate of steplength

    Dc     = np.dot(D,c)
    cDTDc  = np.dot(Dc.transpose(),Dc)
    cTc    = np.dot(c.transpose(),c)
    alpha_last = alpha
    alpha  = cTc/cDTDc

#   Calculate the steplength by a simplified line search

    k_soln = 0
    for k in range(0,n_alpha):
        m_cg_i[k]   = m_cg[iter] - alpha_i[k] * alpha * c
        d_c        = d_(m_cg_i[k])
        Delta_d    = d_c - d_0
        E_cg_i[k]   = np.dot(Delta_d.transpose(),Delta_d)
    for k in range(1,n_alpha-1):
        if E_cg_i[k]<E_cg_i[k-1] and E_cg_i[k]<E_cg_i[k+1]:
            k_soln = k
    alpha          = alpha * alpha_i[k_soln]
    m_cg[iter+1]    = m_cg[iter] - alpha * c

#-----------------------------------------------------
# Newton method iterations
#-----------------------------------------------------

for iter in range(0,n_iter):
    D           = D_(m_h[iter])   # Linearized theory about current model parameters
    d           = d_(m_h[iter])   # Data calculations
    DT          = D.transpose()   # D transpose
    Delta_d     = d - d_0         # Residuals (current error)
    E_h[iter]   = np.dot(Delta_d.transpose(),Delta_d)
    g           = np.dot(DT,np.dot(Cdi,Delta_d))  # Gradient direction

#   Calculate the inverse Hessian

    H           = np.dot(DT,np.dot(Cdi,D))

#   Update the model parameters by the Newton method

    m_h[iter+1] = m_h[iter] - np.dot(inv(H),g)

# Calculate the data at the solution

d_soln = d_(m_h[n_iter-1])



# + {"code_folding": [0]}
# Plots
#----------------------------------------------------------------
# Plot the true data and calculated
# data at the two solutions: steepest
# descent and conjugate gradient
#----------------------------------------------------------------

fig = plt.figure(1, figsize=(15,10))
ax3 = fig.add_subplot(221)
plt.title('Data plots')
ax3.plot(d_true,'g.-')
ax3.plot(d_0 ,'r--')
ax3.plot(d_soln,'b-')
ax3.legend(['d_true','d_observed','d_solution'], borderaxespad=0.) # ,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=3, mode="expand",
plt.xlabel('x', fontsize=14)     # Label x axis
plt.ylabel('d(x)', fontsize=14)  # Label y axis
#plt.savefig('ls_linear_data.png')
#plt.show()
#----------------------------------------------------------------
# Plot the objective function and path to a solution
# as a colour image for the various methods
#----------------------------------------------------------------

# Steepest descent method
plt.subplot(222)
image = plt.imshow(E1_obj, vmax=100, extent=[z_min, z_max, v_min, v_max],cmap='rainbow')
plt.plot(m_sd[:,0],m_sd[:,1],'.-y', ms=8, markeredgewidth=0, lw=2, color='yellow')
plt.title('Objective function: steepest descent method')
plt.colorbar(image)            # Make colorbar
plt.xlabel('z', fontsize=14)   # Label x axis
plt.ylabel('v', fontsize=14)   # Label y axis
#plt.savefig('ls_nonlinear_objective_sd.png')
#plt.show()


# Conjugate-gradient method
plt.subplot(223)
image = plt.imshow(E1_obj, vmax=100, extent=[z_min, z_max, v_min, v_max],cmap='rainbow')
plt.plot(m_cg[:,0],m_cg[:,1],'.-y', ms=8, markeredgewidth=0, lw=2, color='yellow')
plt.title('Objective function: conjugate gradient method')
plt.colorbar(image)            # Make colorbar
plt.xlabel('z', fontsize=14)   # Label x axis
plt.ylabel('v', fontsize=14)   # Label y axis
#plt.savefig('ls_nonlinear_objectivE_cgg.png')
#plt.show()


# Newton method
plt.subplot(224)
image = plt.imshow(E1_obj, vmax=100, extent=[z_min, z_max, v_min, v_max],cmap='rainbow')
plt.plot(m_h[:,0],m_h[:,1],'.-y', ms=8, markeredgewidth=0, lw=2, color='yellow')
plt.title('Objective function: Newton method')
plt.colorbar(image)            # Make colorbar
plt.xlabel('z', fontsize=14)   # Label x axis
plt.ylabel('v', fontsize=14)   # Label y axis
#plt.savefig('ls_nonlinear_objective_newton.png')
plt.subplots_adjust(hspace=0.3)

plt.show()
# -


