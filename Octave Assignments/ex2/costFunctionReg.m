function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

t = sigmoid(X*theta);

p = sigmoid(X*theta);

the_one = ones(length(t), 1);

theta_reg = theta(2:length(theta), 1);

J = (y'*log(t) + (1-y)'*(log(1-t)))*(-1/m) + (theta_reg'*theta_reg)*(lambda/(2*m));

grad = (1/m)*X'*(t-y);

grad = grad + (lambda/m)*theta;

grad(1) = ((the_one)')*(t-y)*(1/m);


% =============================================================

end
