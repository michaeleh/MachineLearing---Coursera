function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
sqrErr = (h-y).^2;
sumOfErr = sum(sqrErr);
tmp_theta = theta;
tmp_theta(1) =0;
sumOfRegular = sum(tmp_theta.^2);
J = sumOfErr/(2*m) + sumOfRegular*lambda/(2*m);

inner_sum_grad = 0;
theta_length = length(theta);
for j=1:theta_length
  sum_result = 0;
  for i=1:length(X(:,1))
    sum_result += (h(i)-y(i))*X(i,j);
  endfor
  grad(j) = sum_result/m;
  
  if j~=1
    grad(j) += (lambda/m) * theta(j);
  endif
endfor









% =========================================================================

grad = grad(:);

end
