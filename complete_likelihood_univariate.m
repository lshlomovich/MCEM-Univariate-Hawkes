function [log_ll] = complete_likelihood_univariate(times, v, alpha, beta, end_time)

% Function to compute the likelihood of univariate times under an
% exponential Hawkes model with parameters given by v, alpha and beta.
% Simulation time given by 'end_time'.

if min(size(times))==1
    % If the times aren't sorted 
	if  any(sort(times)~=times)
    		log_ll = -inf;
    	return
    end
    % If the times have duplicates
	if length(unique(times))<length(times)
    		log_ll = -inf;
    	return
    end
    % If there are times greater than the simulation window
	if max(times>end_time)
    		log_ll = -inf;
    	return
	end
end

% Vectorised version
A_s = zeros(size(times));
A_s(:,1) = 0;
tp_diff = diff(times,1,2);
[~,total_events] = size(times);
for tp = 1:total_events-1
    A_s(:,tp+1) = (exp(-beta*tp_diff(:,tp))).*(1+A_s(:,tp));
end

% Log likelihood
log_ll = -v*end_time + (alpha/beta)*sum(exp(-beta*(end_time-times))-1,2) + sum(log(v + alpha*A_s),2);

end
