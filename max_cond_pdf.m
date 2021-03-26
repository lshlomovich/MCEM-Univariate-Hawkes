function [pdf_val,interim_pdf_val] = max_cond_pdf(miss_times, history, bin_width, max_bin, v, alpha, beta)
% This function computes the likelihood of the current bin given the
% history of the process. That is, with the parameters v,alpha,beta, and
% the history of the process, it can take proposed missing times and
% compute their likelihood. Thus, maximising this function allows us to
% resdistribute points. Note that we truncate this likelihood to ensure it
% sits between the bin bounds.

% m events in bin to be simulated
% miss_times is row vector of missing time points 

m = length(miss_times);
if any(sort(miss_times)~=miss_times)
    pdf_val = inf;
    return
end

if (min(miss_times)<=(max_bin-bin_width) || max(miss_times)>=max_bin)
    pdf_val = inf;
    return
end

if length(unique(miss_times))~=m
    pdf_val = inf;
    return
end

all_events = [0;history;miss_times'];

point_n = length(history)+1;
product_term = 1;
for i=1:m 
    first_product = v + alpha*sum(exp(-beta*(all_events(point_n+i)-all_events(2:point_n+i-1))));
    second_product = exp(alpha/beta * sum( exp(-beta*(all_events(point_n+i)-all_events(2:point_n+i-1))) - exp(-beta*(all_events(point_n+i-1)-all_events(2:point_n+i-1)))));
    product_term = product_term * first_product * second_product;
end
pdf_val = product_term*exp(-v*(all_events(end)-all_events(point_n)));
interim_pdf_val = pdf_val;

% TRUNCATION METHOD 
upper_cdf = 1;
ub_times = [0;history;transpose(miss_times(1:end-1));max_bin];
middle_prod = 1;
for i=1:m
    ith_prod = 1 - exp(-v*(ub_times(point_n+i)-ub_times(point_n+i-1)))*exp(alpha/beta * (sum(exp(-beta*(ub_times(point_n+i)-ub_times(2:point_n+i-1)))) - sum(exp(-beta*(ub_times(point_n+i-1)-ub_times(2:point_n+i-1)))))); % same as OG
    if (i>=2 && i <= m-1)
        middle_prod = middle_prod * ith_prod;
    end
    upper_cdf = ith_prod * upper_cdf;
end
lb_times = [0;history;transpose(miss_times(1:end))];
final_cdf_val = 1 - exp(-v*(lb_times(point_n+m)-lb_times(point_n+m-1)))*exp(alpha/beta * (sum(exp(-beta*(lb_times(point_n+m)-lb_times(2:point_n+m-1)))) -sum(exp(-beta*(lb_times(point_n+m-1)-lb_times(2:point_n+m-1))))));
first_cdf_val = 1 - exp(-v*(max_bin-bin_width-lb_times(point_n)))*exp(alpha/beta * (sum(exp(-beta*(max_bin-bin_width-lb_times(2:point_n)))) -sum(exp(-beta*(lb_times(point_n)-lb_times(2:point_n))))));
lower_cdf = middle_prod * final_cdf_val * first_cdf_val;

if upper_cdf-lower_cdf ~= 0
    conditional_pdf_val = pdf_val/(upper_cdf-lower_cdf); 
else
    disp('0 for max_cond_pdf denom')
    conditional_pdf_val = pdf_val;
end

pdf_val = -conditional_pdf_val;


end




