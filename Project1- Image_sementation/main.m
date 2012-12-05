%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PROJECT 1- Image Segmentation using Neural Networks
% Course - ECE8493 - Introduction to Neural Networks
%
% author@ Ankit Arya 
% Undergraduate student, CSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Read original image
I=imread('flowers.jpg');

%Create dataset for neural network
X= [reshape(I(21:30,21:30,:),100,3);reshape(I(61:70,41:50,:),100,3);reshape(I(61:70,51:60,:),100,3);reshape(I(81:90,71:80,:),100,3);reshape(I(91:100,41:50,:),100,3);reshape(I(91:100,61:70,:),100,3); reshape(I(1:10,91:120,:),300,3)];

X= double(X);



%neural networks parameters
input_layer_size  =3 ;  % 104x82 Input Images
hidden_layer_size = 3;   % 3 hidden units
num_labels = 3;          % 3 labels, from 1 to 10   
y= [ ones(300,1); 2*ones(300,1); 3*ones(300,1)]; 
skip_training=true;



if ~skip_training,
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);

initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


fprintf('\nTraining Neural Network... \n')

%change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 300);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause; 
end

[pred, soft_class]= predict(5*Theta1, 5*Theta2, X);
%{
I=imread('flowers.jpg');
[m,n,s]=size(I);
testData= double(reshape(I,m*n,3));

[pred, soft_class]= predict(Theta1, Theta2, testData);
im= reshape(pred,m,n);
thresh=2;


% view flowers
binaryImage= im < thresh;
%binaryImage= imfill(binaryImage,'holes');
imwrite(binaryImage,'flower','png');

%leaves
binaryImage2= [im == thresh];
%binaryImage2= imfill(binaryImage2,'holes');
imwrite(binaryImage2,'leaves','png');

%bckground
df= [im >2];
%df= imfill(df,'holes');
imwrite(df,'bckgrnd','png');

save 'hard_map.mat' pred
save 'soft_map.mat' soft_class
%}
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
