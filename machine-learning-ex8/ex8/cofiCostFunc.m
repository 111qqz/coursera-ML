function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%num_movies  % 4
%num_features % 3
%num_users    % 5
size(X); % 4*3
size(Y); % 4*5
size(Theta); %5*3
size(R);   % 4*5
%X*Theta'
%(X*Theta').*R
tmp = ((X*Theta').*R-Y).^2;
J = sum(sum(tmp))/2;
reg = (sum(sum(Theta.^2)) + sum(sum(X.^2))) *lambda / 2;
J = J + reg;

% X_grad = ((X*Theta').*R-Y)*Theta;

for i = 1:num_movies
    idx = find(R(i,:)==1);
    Theta_tmp = Theta(idx,:);
    Y_tmp = Y(i,idx);
    size(Theta_tmp);
    size(Y_tmp);
    X_grad (i, :) = (X(i, :)*Theta_tmp' - Y_tmp ) * Theta_tmp + lambda * X(i,:);
    size(X_grad(i,:));
endfor 

for i = 1:num_users
    idx = find(R(:,i)==1);
    X_tmp = X(idx,:);
    Y_tmp = Y(idx,i);
    % Theta_grad should be 1*3
    Theta_grad(i,:) = (X_tmp*Theta(i,:)' - Y_tmp )' * X_tmp + lambda * Theta(i,:);
endfor










X;
size(X);
X_grad;
size(X_grad);




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
