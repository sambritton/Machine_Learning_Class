function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m,1), X];

%h_theta1 = sigmoid(theta*a) for theta_1 and theta_2
z1 = X*Theta1';
h1 = sigmoid(z1);
h1 = [ones(m,1), h1];

%h2 represents the probability that the given data is in a certain
%category. e.g. if the highest probability is in index 5, then 5 is chosen
%as the category for the image.
z2 = h1*Theta2';
h2 = sigmoid(z2);

%The maximum acts along axis 2 and returns h2 in pval and the index in p.
%The value of h2 represents the probability of correct choice and the p is 
[pval, p] = max(h2, [], 2);





% =========================================================================


end
