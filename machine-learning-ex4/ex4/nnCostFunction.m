function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
% Convert y from (1-10) class into num_labels vector. This makes it so that
% y=5 turns into y = (0,..1,...0) with one at index 5.
yd = eye(num_labels);
y = yd(y,:);

%In this section, we build up layer by layer. Make sure to add the row of
%zeros after moving up each row.
%%% Map from Layer 1 to Layer 2
a1=X;
% Coverts to matrix of 5000 examples x 26 thetas
z2=X*Theta1';
% Sigmoid function converts to p between 0 to 1
a2=sigmoid(z2);

%%% Map from Layer 2 to Layer 3
% Add ones to the h1 data matrix
a2=[ones(m, 1) a2];
% Converts to matrix of 5000 exampls x num_labels 
z3=a2*Theta2';
% Sigmoid function converts to p between 0 to 1
a3=sigmoid(z3);
% Compute cost
logisf=(-y).*log(a3)-(1-y).*log(1-a3); 

%% Regularized cost
%these two thetas are used for the nonlinear terms. 
Theta1s=Theta1(:,2:end);
Theta2s=Theta2(:,2:end);
%total cost. Make sure you use two sums when feeding in a matrix
J=((1/m).*sum(sum(logisf)))+(lambda/(2*m)).*(sum(sum(Theta1s.^2))+sum(sum(Theta2s.^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Set all the D to zeros
sumdelta_1=0;
sumdelta_2=0;

% Compute delta, tridelta and big D
%First we compute the difference between the network activation (a3) and
%the actual target (y). 
	delta_3=a3-y;
    
    %pad z2 for multiplication. Compute delta2. Take all but the top one.
    z2=[ones(m,1) z2];
	delta_2=delta_3*Theta2.*sigmoidGradient(z2);
    delta_2=delta_2(:,2:end);
    
    %sum errors
	sumdelta_1=sumdelta_1+delta_2'*a1; % Same size as Theta1_grad (25x401)
    sumdelta_2=sumdelta_2+delta_3'*a2; % Same size as Theta2_grad (10x26)
    
    %determine unregularized gradient for nn cost function.
	Theta1_grad=(1/m).*sumdelta_1;
    Theta2_grad=(1/m).*sumdelta_2;
    %Theta1_grad=0;
	%Theta2_grad=0;
%end
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%regularize theta. 

Theta1_grad(:, 2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:, 2:end);













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
