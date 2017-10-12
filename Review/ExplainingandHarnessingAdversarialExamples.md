– Summary:

This paper pushes the research of adversarial examples and further improves
training process of most commonly used linear neural networks. It argues that
the adversarial issues come from the linear part of the neural network, instead
of notorious non-linearity and overfitting of deep neural networks. It proposes
a constructive method to find adversarial examples for simple and more generally
used neural networks, called fast gradient sign method. And this method can also
treated as a kind of regularization added to the total cost. In the end, several
quantitative experiments back the above arguments.

– List of positive points/Strengths:

This paper initialize all argument from analysis on linear explanation of
adversarial examples. The trick behind adversarial examples is the precision of
input. When input is perturbed a little bit, the max norm of the deviation
doesn't grow linearly with the number of dimensions, but the overall change in
activation function does grow linearly with the number of dimension. This means
that for a very large number of hidden units in certain hidden layer, the
overall activation can easily jump to its neighboring area, which will render a
different and wrong class.

From above analysis, the authors propose a systematic and cheap way to generate
adversarial examples. They show that for logistic regression, fast gradient sign
method is more or less like L1 regularization. The difference is that the
penalties are applied to activation function directly rather than as a separate
term of final loss function. This tells us only adding L1 regularization in the
end will generally over-estimate adversarial impact and make trained networks
vulnerable to adversarial examples.


To answer the question of why adversarial examples generalize across different
networks and data set, the authors argue that by tracing out different values of 
ε we see that adversarial examples occur in contiguous regions of the 1-D 
subspace defined by the fast gradient sign method, not in fine pockets. The
direction of adversarial examples matters, instead of concrete points.

– List of negative points/Weaknesses:

The paper mentions that it wants to maximize w*x_prime given the constraint on
the max norm of eta. But symmetrically speaking, it can also optimize in the
other direction, i.e. Minimizing the objective given the constaint. I think the
complete formulation should be maximize the norm of w*x_prime. The solution can
still be sign function of w but with a different sign.

The paper argues that fast gradient method is actually add L1 penalties inside
activation function. But later on, they say that they can construct a brand new
loss function based on fast gradient method. I don't get the transition from
inside activation function regularization to the final loss function
regularization.

– Reflections

This paper is a great paper, discovering some new stuff of adversarial examples.
It also helps to boost the training of networks, making it more robust to easily
mis-classified examples.
