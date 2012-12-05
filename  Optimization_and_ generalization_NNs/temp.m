W1= model.input_to_hid;
W2= model.hid_to_class;
a1=data.inputs;
y=data.targets;
lambda=wd_coefficient;

z1 = W1 * a1; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
a2 = logistic(z1); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
z2 = W2* a2; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
%a3= logistic(z2);
%hx=a3;
class_input=z2;
class_normalizer = log_sum_exp_over_rows(z2); % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
 class_prob = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>
a3=class_prob;
 % classification_loss = -mean(sum(log_class_prob .* data.targets, 1));


delta3= a3- y;

delta2= (W2'*delta3).*a3.*(1-a3);
Delta1= delta2*a1';
Delta2=delta3*a2';

m= size(a1,2);

Theta1_grad=Delta1/m +  lambda*W1;
%Theta1_grad(:,2:end)=Theta1_grad(:,2:end) + lambda*W1/m;
Theta2_grad=Delta2/m + lambda*W2;	 	
%Theta2_grad(:,2:end)=Theta2_grad(:,2:end) + lambda*W2/m;
