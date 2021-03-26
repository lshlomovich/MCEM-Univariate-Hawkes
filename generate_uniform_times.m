function [times_cell, log_density] = generate_uniform_times(data, p, bin_width)

%This function samples times from independent uniform distributions of
%non-empty bins 
% p is the dimension of the process

%This doesn't account for different bin sizes so if you choose to use this
%you'll need to adjust the code accordingly. 
 
% Joint distribution of the order statistics from an absolutely cts dist. 
% n!f_X(x_1)...f_X(x_n), where x_1 <= ... <= x_n
% So for uniform it is n!1/(bin_width)^n (check if 1/(bin_width)^n is included?)
% This is for each bin so we have n_1!....n_{n_bins}!
log_density = sum(sum(log(factorial(data)))) - sum(data)*log(bin_width);   

for j=1:p
    times = [];
    non_empty_bins = find(data(:,j));
    for i=1:length(non_empty_bins)
        counts = data(non_empty_bins(i),j); 
    
        if counts == 1 
            time = bin_width*(non_empty_bins(i)-1) + bin_width*rand;
        else 
           time = bin_width*(non_empty_bins(i)-1) + bin_width*sort(rand(1,counts));

        end 
    
        times = [times time];       
        
    end 
    if p > 1
        times_cell{j} = times;
    else
        times_cell = times;
    end
end


end 

            