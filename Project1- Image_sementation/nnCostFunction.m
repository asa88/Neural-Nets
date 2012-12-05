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



% Theta1 is 25 401 Theta2 is 10 26
X = [ones(m, 1) X]; %5000 401
a1=X;
z2= Theta1*a1';  % is 25 5000
a2=sigmoid(z2);
a2=[ones(1,m);a2];  %is 26 5000
z3=(Theta2*a2)';     %is5000 10
a3=sigmoid(z3);
%a3=[ones(1,m);a3]; 
hx=a3;
I=eye(3);
y=I(y,:);


% hx size is 5000 10 , y 5000 10
temp1=Theta1;
temp1(:,1)=[];
temp2=Theta2;
temp2(:,1)=[];

const=(lambda*(sum(sum(temp1.^2)) + sum(sum(temp2.^2))))/2;
for i=1:m,
    J=J+(-y(i,:)*log(hx(i,:))'-((1-y(i,:))*log(1-hx(i,:))'));
end
J=(J+const)/m;


% back propogation algorithm...

%a2 is 26 5000, a3 is 5000 10, Theta2 26 10
Delta1=0;
Delta2=0;
for t=1:m, 
  a_1=X(t,:); % 1 401
  z_2= Theta1*a_1'; % 25 1
  a_2=sigmoid(z_2); 
  a_2=[1;a_2]; %26 1
  z_2=[1;z_2];
  z_3=(Theta2*a_2)'; % 1 10
  a_3=sigmoid(z_3); % 1 10
  hx=a_3;
  delta3=a_3-y(t,:); % y is 1 10 so is a3
  delta2=Theta2'*delta3'.*sigmoidGradient(z_2);% 26 1
  delta2=delta2(2:end);
  Delta1= Delta1 + delta2*a_1; % 25 401 
  Delta2= Delta2 + delta3'*a_2'; % 10 26
end

Theta1_grad=Delta1/m;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end) + lambda*Theta1(:,2:end)/m;
Theta2_grad=Delta2/m;	 	
Theta2_grad(:,2:end)=Theta2_grad(:,2:end) + lambda*Theta2(:,2:end)/m;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
