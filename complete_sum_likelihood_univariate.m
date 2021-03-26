function log_lik = complete_sum_likelihood_univariate(times, v,alpha,beta, end_time,old_weights,p)

% This is the Q function of the EM algorithm to be numerically
% maximised at each iteration.

% times:            sampled time stamps
% v, alpha, beta:   are the parameters to be used
% end_time:         end of the simulation window
% old_weights:      the importance sampling (IS) weights to be used
% p:                is the dimension of the Hawkes process

     if v <= 0
         log_lik = inf;
         %disp('v: negative baseline')
         return
     end
     if min(min(alpha)) < 0 
         log_lik = inf;
         %disp('A: negative excitation matrix')
         return
     end
     if any(any(alpha>=beta)) 
         log_lik = inf;
         %disp('A greater than B')
         return
     end
     if det(eye(p) - alpha./beta) <= 0
         log_lik = inf;
         %disp('1) A./B: determinant less than zero')
         return
     end
     %Spectral radius of Gamma = \int_0^inf G(u)du is < 1.
     if any((eye(p) - alpha./beta)\v < 0)  
         log_lik = inf;
         %disp('2) A./B: determinant less than zero')
         return
     end

% N = size(times,1); 
% log_likelihoods = 1:N; 
% Unvectorised version:
%for i=1:N 
%    log_likelihoods(i) = complete_likelihood(times(i,:), v,alpha,beta, end_time)*old_weights(i); 
%end 

% Vectorised version:
log_likelihoods = complete_likelihood_univariate(times, v, alpha, beta, end_time);
if size(log_likelihoods) ~= size(old_weights)
	old_weights = old_weights';
end
log_likelihoods = log_likelihoods.*old_weights;

% Take the mean of the weighted log-likelihoods
log_lik = -mean(log_likelihoods); 
end 
