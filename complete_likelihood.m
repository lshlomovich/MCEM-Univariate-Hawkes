function [log_ll, grad, hessian] = complete_likelihood(times, v, a, b, end_time, p)

% times:           the time stamps 
% Note, in the univariate case this has been vectorised so it is expecting
% a row vector of times, or a matrix where each row is an MC rep
% In the multivariate case, this will take an Nxp matrix (N:number of bins)
% v, a, b:         nu, alpha, beta, parameters of exponential kernel
% end time:        max simulation time
% p:               dimension of the data

% log_ll is the NEGATIVE log likelihood %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialise a gradient and hessian
grad = -1e10*ones(size([v,a,b]));
hessian = -1e10*ones(numel([v,a,b]),numel([v,a,b]));

% Conditions for stationarity
if length(v)>1 && eigs(a./b, 1, 'lm')-1 >= 0
    log_ll = inf; % set neg ll to inf
    %disp('non-stationary')
elseif length(v) == 1 && a>b
    log_ll = inf; % set neg ll to inf
    %disp('non-stationary')
end


%% Vectorising.
if p==1
    A_s = zeros(size(times)); 
    A_s(:,1) = 0; % set the first value as zero in all cases
    B_s = zeros(size(times,1),1); 
    C_s = zeros(size(times,1),1); 
    tp_diff = diff(times,1,2);  % get the differences for each row 
    [~,total_events] = size(times);
    
    for tp = 1:total_events-1
        A_s(:,tp+1) = (exp(-b*tp_diff(:,tp))).*(1+A_s(:,tp));
        B_s = [B_s, sum((times(:,tp+1)-times(:,1:tp)).*exp(-b*(times(:,tp+1)-times(:,1:tp))),2)];
        C_s = [C_s, sum(((times(:,tp+1)-times(:,1:tp)).^2).*exp(-b*(times(:,tp+1)-times(:,1:tp))),2)];
    end
    % As given in Da Fonseca and Zaatour, Hawkes process: fast calibration,
    % application to trade clustering and diffusive limit
    % log_ll vector for each of the m Monte Carlo samples
    log_ll = -v*end_time + (a/b)*sum(exp(-b*(end_time-times))-1,2) + sum(log(v + a*A_s),2); 
    % alternative: https://uk.mathworks.com/matlabcentral/fileexchange/68936-log-likelihood-of-the-hawkes-process
    
    log_ll = -log_ll; % NEGATIVE LL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    exp_term = exp(-b*(end_time-times)); % yields matrix of same size as time_diffs
    gradient_baseline = -end_time + sum(1./(v+a*A_s),2);  
    gradient_A = sum((1/b)*(exp_term-1),2) + sum(A_s./(v+a*A_s),2);
    gradient_B = -a*(sum( (1/b)*(end_time-times).*exp_term + (1/(b^2))*(exp_term-1),2) ) - sum(a*B_s./(v+a*A_s),2);
    grad = -[gradient_baseline'; gradient_A'; gradient_B'];
    
    % HESSIAN
    nu_nu = sum(-1./((v+a*A_s).^2),2);
    alpha_alpha = - sum((A_s./(v+a*A_s)).^2,2);
    beta_beta = a* sum( (1/b)*((end_time-times).^2).*exp_term + (2/(b^2))*(end_time-times).*exp_term + (2/(b^3))*(exp_term-1), 2) + sum( a*C_s./(v+a*A_s) - (a*B_s./(v+a*A_s)).^2, 2 ); 
    alpha_beta = - sum( (1/b)*(end_time-times).*exp_term + (1/(b^2))*(exp_term-1), 2) + sum( -B_s./(v+a*A_s) + (a*A_s.*B_s)./((v+a*A_s).^2), 2 );  
    beta_alpha = alpha_beta;
    alpha_nu = sum(-A_s./((v+a*A_s)).^2,2);
    nu_alpha = alpha_nu;
    beta_nu = sum(a*B_s./((v+a*A_s).^2),2);
    nu_beta = beta_nu; 
    
    hr1 = [nu_nu, nu_alpha, nu_beta];
    hr2 = [alpha_nu, alpha_alpha, alpha_beta];
    hr3 = [beta_nu, beta_alpha, beta_beta];
    
    hessian = -[hr1;hr2;hr3];
    
% MULTIVARIATE VERSIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif p==2 % unvectorised - ie in complete_sum_liklihood we loop over samples
    % Brute force coding the bivariate case
    branch_ratio = a./b;
    times1 = times{1}';
    times2 = times{2}';
    
    A_11 = zeros(length(times1),1); A_11(1) = 0;
    A_12 = zeros(length(times1),1); A_12(1) = 0;
    A_21 = zeros(length(times2),1); A_21(1) = 0;
    A_22 = zeros(length(times2),1); A_22(1) = 0;
    B_11 = 0; B_12 = 0; B_21 = 0; B_22 = 0; 
    C_11 = 0; C_12 = 0; C_21 = 0; C_22 = 0; 
    % A_11 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tp_diff1 = diff(times1);
    for tp = 1:length(times1)-1
        A_11(tp+1) = (exp(-b(1,1)*tp_diff1(tp)))*(A_11(tp) + 1);
        B_11 = [B_11; sum((times1(tp+1)-times1(1:tp)).*exp(-b(1,1)*(times1(tp+1)-times1(1:tp))))];
        C_11 = [C_11; sum(((times1(tp+1)-times1(1:tp)).^2).*exp(-b(1,1)*(times1(tp+1)-times1(1:tp))))];
    end
    % A_22 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tp_diff2 = diff(times2);
    for tp = 1:length(times2)-1
        A_22(tp+1) = (exp(-b(2,2)*tp_diff2(tp)))*(A_22(tp) + 1);
        B_22 = [B_22; sum((times2(tp+1)-times2(1:tp)).*exp(-b(2,2)*(times2(tp+1)-times2(1:tp))))];
        C_22 = [C_22; sum(((times2(tp+1)-times2(1:tp)).^2).*exp(-b(2,2)*(times2(tp+1)-times2(1:tp))))];
    end
    % A_12 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp1 = 2:length(times1) %for tp = 1:length(times1)-1
        j_proc_times = intersect((times2(times2 < times1(tp1))),(times2(times2 >= times1(tp1-1))));
        A_12(tp1) = (exp(-b(1,2)*tp_diff1(tp1-1)))*A_12(tp1-1) + sum(exp(-b(1,2)*(times1(tp1) - j_proc_times)));
        times_set1 = [];
        times_set1 = times2(times2 < times1(tp1));
        B_12 = [B_12; sum((times1(tp1)-times_set1).*exp(-b(1,2)*(times1(tp1)-times_set1)))];
        C_12 = [C_12; sum(((times1(tp1)-times_set1).^2).*exp(-b(1,2)*(times1(tp1)-times_set1)))];
    end
    % A_21 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp2 = 2:length(times2)  %for tp = 1:length(times2)-1
        j_proc_times = intersect((times1(times1 < times2(tp2))),(times1(times1 >= times2(tp2-1))));
        A_21(tp2) = (exp(-b(2,1)*tp_diff2(tp2-1)))*A_21(tp2-1) + sum(exp(-b(2,1)*(times2(tp2) - j_proc_times)));
        times_set2 = [];
        times_set2 = times1(times1 < times2(tp2));
        B_21 = [B_21; sum((times2(tp2)-times_set2).*exp(-b(2,1)*(times2(tp2)-times_set2)))];
        C_21 = [C_21; sum(((times2(tp2)-times_set2).^2).*exp(-b(2,1)*(times2(tp2)-times_set2)))];
    end
    
    ll1_term1 = branch_ratio(1,1)*sum(exp(-b(1,1)*(end_time-times1))-1) + branch_ratio(1,2)*sum(exp(-b(1,2)*(end_time-times2))-1); 
    ll2_term1 = branch_ratio(2,2)*sum(exp(-b(2,2)*(end_time-times2))-1) + branch_ratio(2,1)*sum(exp(-b(2,1)*(end_time-times1))-1); 
    %    
    ll1_term2 = sum(log(v(1) + a(1,1)*A_11 + a(1,2)*A_12));
    ll2_term2 = sum(log(v(2) + a(2,2)*A_22 + a(2,1)*A_21));
    %
    log_ll1 = -v(1)*end_time + ll1_term1 + ll1_term2;
    log_ll2 = -v(2)*end_time + ll2_term1 + ll2_term2;
    
    log_ll = log_ll1 + log_ll2;
    
    log_ll = -log_ll; 
    exp_term11 = exp(-b(1,1)*(end_time-times1)); 
    exp_term12 = exp(-b(1,2)*(end_time-times2)); 
    exp_term21 = exp(-b(2,1)*(end_time-times1)); 
    exp_term22 = exp(-b(2,2)*(end_time-times2)); 
    denom_term1 = sum(1./(v(1)+a(1,1)*A_11 + a(1,2)*A_12)); 
    denom_term2 = sum(1./(v(2)+a(2,2)*A_22 + a(2,1)*A_21));

    gradient_baseline1 = -end_time + denom_term1;  
    gradient_baseline2 = -end_time + denom_term2;  
    grads_base = [gradient_baseline1;gradient_baseline2];

    gradient_A11 = sum((1/b(1,1))*(exp_term11-1)) + sum((A_11)./(v(1)+a(1,1)*A_11+a(1,2)*A_12)); % first term equiv to (1/b(1,1))*(A_11(end)-(n_arr1-1))
    gradient_A12 = sum((1/b(1,2))*(exp_term12-1)) + sum((A_12)./(v(1)+a(1,1)*A_11+a(1,2)*A_12)); % first term equiv to (1/b(1,2))*(A_12(end)-(n_arr2-1))
    gradient_A21 = sum((1/b(2,1))*(exp_term21-1)) + sum((A_21)./(v(2)+a(2,1)*A_21+a(2,2)*A_22)); % (1/b(2,1))*(A_21(end)-(n_arr1-1)) 
    gradient_A22 = sum((1/b(2,2))*(exp_term22-1)) + sum((A_22)./(v(2)+a(2,2)*A_22+a(2,1)*A_21)); % (1/b(2,2))*(A_22(end)-(n_arr2-1))
    gradsA = [gradient_A11;gradient_A21;gradient_A12;gradient_A22];

    gradient_B11 = -a(1,1)*(sum( (1/b(1,1))*(end_time-times1).*exp_term11 + (1/(b(1,1)^2))*(exp_term11-1))) - sum((a(1,1)*B_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12));
    gradient_B12 = -a(1,2)*(sum( (1/b(1,2))*(end_time-times2).*exp_term12 + (1/(b(1,2)^2))*(exp_term12-1))) - sum((a(1,2)*B_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12));
    gradient_B21 = -a(2,1)*(sum( (1/b(2,1))*(end_time-times1).*exp_term21 + (1/(b(2,1)^2))*(exp_term21-1))) - sum((a(2,1)*B_21)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22));
    gradient_B22 = -a(2,2)*(sum( (1/b(2,2))*(end_time-times2).*exp_term22 + (1/(b(2,2)^2))*(exp_term22-1))) - sum((a(2,2)*B_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22));
    gradsB = [gradient_B11;gradient_B21;gradient_B12;gradient_B22];

    grad = -[grads_base; gradsA; gradsB];
        
    % HESSIAN 
    % CASE 1: \frac{\partial L^m}{\partial nu_m^2} 
    nu_1_nu_1 = - sum(1./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    nu_2_nu_2 = - sum(1./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    % CASE 2: \frac{\partial L^m}{\partial nu_m \partial nu_m} 
    nu_1_nu_2 = 0;
    nu_2_nu_1 = 0;
    % CASE 3: \frac{\partial L^m}{\partial \alpha_{mn}^2} 
    alpha_11_alpha_11 = - sum(((A_11)./(v(1)+a(1,1)*A_11+a(1,2)*A_12)).^2); 
    alpha_12_alpha_12 = - sum(((A_12)./(v(1)+a(1,1)*A_11+a(1,2)*A_12)).^2);
    alpha_21_alpha_21 = - sum(((A_21)./(v(2)+a(2,1)*A_21+a(2,2)*A_22)).^2);
    alpha_22_alpha_22 = - sum(((A_22)./(v(2)+a(2,2)*A_22+a(2,1)*A_21)).^2);
    % CASE 4: \frac{\partial L^m}{\partial \alpha_{mn} \partial \alpha_{mn'}} 
    alpha_11_alpha_12 = - sum((A_11.*A_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    alpha_12_alpha_11 = alpha_11_alpha_12;
    alpha_21_alpha_22 = - sum((A_21.*A_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    alpha_22_alpha_21 = alpha_21_alpha_22;
    % CASE 5: \frac{\partial L^m}{\partial \alpha_{mn} \partial \alpha_{m'n'}} 
    alpha_11_alpha_22 = 0;
    alpha_11_alpha_21 = 0;
    alpha_12_alpha_21 = 0;
    alpha_12_alpha_22 = 0;
    alpha_21_alpha_12 = 0;
    alpha_21_alpha_11 = 0;
    alpha_22_alpha_11 = 0;
    alpha_22_alpha_12 = 0;
    % CASE 6: \frac{\partial L^m}{\partial \alpha_{mn} \partial \nu_m} 
    alpha_11_nu_1 = - sum((A_11)./(v(1)+a(1,1)*A_11+a(1,2)*A_12).^2);
    nu_1_alpha_11 = alpha_11_nu_1;
    alpha_12_nu_1 = - sum((A_12)./(v(1)+a(1,1)*A_11+a(1,2)*A_12).^2);
    nu_1_alpha_12 = alpha_12_nu_1;
    alpha_21_nu_2 = - sum((A_21)./(v(2)+a(2,1)*A_21+a(2,2)*A_22).^2);
    nu_2_alpha_21 = alpha_21_nu_2;
    alpha_22_nu_2 = - sum((A_22)./(v(2)+a(2,2)*A_22+a(2,1)*A_21).^2);
    nu_2_alpha_22 = alpha_22_nu_2; 
    % CASE 7: \frac{\partial L^m}{\partial \alpha_{mn} \partial \nu_m'} 
    alpha_11_nu_2 = 0; nu_2_alpha_11 = 0;
    alpha_12_nu_2 = 0; nu_2_alpha_12 = 0;
    alpha_21_nu_1 = 0; nu_1_alpha_21 = 0;
    alpha_22_nu_1 = 0; nu_1_alpha_22 = 0;
    % CASE 8: \frac{\partial L^m}{\partial \beta_{mn} \partial \nu_m} 
    beta_11_nu_1 = sum((a(1,1)*B_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    nu_1_beta_11 = beta_11_nu_1; 
    beta_12_nu_1 = sum((a(1,2)*B_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    nu_1_beta_12 = beta_12_nu_1;
    beta_21_nu_2 = sum((a(2,1)*B_21)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    nu_2_beta_21 = beta_21_nu_2;
    beta_22_nu_2 = sum((a(2,2)*B_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    nu_2_beta_22 = beta_22_nu_2;
    % CASE 9: \frac{\partial L^m}{\partial \beta_{mn} \partial \nu_m'}
    beta_11_nu_2 = 0; nu_2_beta_11 = 0; 
    beta_12_nu_2 = 0; nu_2_beta_12 = 0; 
    beta_21_nu_1 = 0; nu_1_beta_21 = 0; 
    beta_22_nu_1 = 0; nu_1_beta_22 = 0; 
    % CASE 10: \frac{\partial L^m}{\partial \beta_{mn} \partial
    % \alpha_{mn}} 
    beta_11_alpha_11 = -(sum( (1/b(1,1))*(end_time-times1).*exp_term11 + (1/(b(1,1)^2))*(exp_term11-1))) - sum((B_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12)) + sum((a(1,1)*B_11.*A_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    alpha_11_beta_11 = beta_11_alpha_11;
    beta_12_alpha_12 = -(sum( (1/b(1,2))*(end_time-times2).*exp_term12 + (1/(b(1,2)^2))*(exp_term12-1))) - sum((B_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12)) + sum((a(1,2)*B_12.*A_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    alpha_12_beta_12 = beta_12_alpha_12;
    beta_21_alpha_21 = -(sum( (1/b(2,1))*(end_time-times1).*exp_term21 + (1/(b(2,1)^2))*(exp_term21-1))) - sum((B_21)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22)) + sum((a(2,1)*B_21.*A_21)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    alpha_21_beta_21 = beta_21_alpha_21;
    beta_22_alpha_22 = -(sum( (1/b(2,2))*(end_time-times2).*exp_term22 + (1/(b(2,2)^2))*(exp_term22-1))) - sum((B_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22)) + sum((a(2,2)*B_22.*A_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    alpha_22_beta_22 = beta_22_alpha_22; 
    % CASE 11: \frac{\partial L^m}{\partial \beta_{mn} \partial \alpha_{mn'}}
    beta_11_alpha_12 = sum((a(1,1)*B_11.*A_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    alpha_12_beta_11 = beta_11_alpha_12;
    beta_12_alpha_11 = sum((a(1,2)*B_12.*A_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12).^2);
    alpha_11_beta_12 = beta_12_alpha_11;
    beta_21_alpha_22 = sum((a(2,1)*B_21.*A_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    alpha_22_beta_21 = beta_21_alpha_22;
    beta_22_alpha_21 = sum((a(2,2)*B_22.*A_21)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22).^2);
    alpha_21_beta_22 = beta_22_alpha_21;     
    % CASE 12:
    beta_11_alpha_22 = 0; alpha_22_beta_11 = 0; 
    beta_11_alpha_21 = 0; alpha_21_beta_11 = 0; 
    beta_12_alpha_21 = 0; alpha_21_beta_12 = 0; 
    beta_12_alpha_22 = 0; alpha_22_beta_12 = 0;
    beta_21_alpha_12 = 0; alpha_12_beta_21 = 0; 
    beta_21_alpha_11 = 0; alpha_11_beta_21 = 0;
    beta_22_alpha_11 = 0; alpha_11_beta_22 = 0;
    beta_22_alpha_12 = 0; alpha_12_beta_22 = 0;
    % CASE 13: \frac{\partial L^m}{\partial \beta_{mn}^2} 
    beta_11_beta_11 = (sum( (2*a(1,1)/(b(1,1)^3))*(exp_term11-1) + (2*a(1,1)/(b(1,1)^2))*(end_time-times1).*exp_term11 + (a(1,1)/(b(1,1)))*((end_time-times1).^2).*exp_term11 )) + sum( ((a(1,1)*C_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12)) - ((a(1,1)*B_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12)).^2);
    beta_12_beta_12 = (sum( (2*a(1,2)/(b(1,2)^3))*(exp_term12-1) + (2*a(1,2)/(b(1,2)^2))*(end_time-times2).*exp_term12 + (a(1,2)/(b(1,2)))*((end_time-times2).^2).*exp_term12 )) + sum( ((a(1,2)*C_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12)) - ((a(1,2)*B_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12)).^2);
    beta_21_beta_21 = (sum( (2*a(2,1)/(b(2,1)^3))*(exp_term21-1) + (2*a(2,1)/(b(2,1)^2))*(end_time-times1).*exp_term21 + (a(2,1)/(b(2,1)))*((end_time-times1).^2).*exp_term21 )) + sum( ((a(2,1)*C_21)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22)) - ((a(2,1)*B_21)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22)).^2);
    beta_22_beta_22 = (sum( (2*a(2,2)/(b(2,2)^3))*(exp_term22-1) + (2*a(2,2)/(b(2,2)^2))*(end_time-times2).*exp_term22 + (a(2,2)/(b(2,2)))*((end_time-times2).^2).*exp_term22 )) + sum( ((a(2,2)*C_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22)) - ((a(2,2)*B_22)./(v(2)+a(2,1)*A_21 + a(2,2)*A_22)).^2);
    % CASE 14: \frac{\partial L^m}{\partial \beta_{mn} \partial \beta_{mn'}}
    beta_11_beta_12 = - sum((a(1,1)*B_11*a(1,2).*B_12)./(v(1)+a(1,1)*A_11+a(1,2)*A_12).^2);
    beta_12_beta_11 = beta_11_beta_12; 
    beta_21_beta_22 = - sum((a(2,1)*B_21*a(2,2).*B_22)./(v(2)+a(2,1)*A_21+a(2,2)*A_22).^2);
    beta_22_beta_21 = beta_21_beta_22;
    % CASE 15: \frac{\partial L^m}{\partial \beta_{mn} \partial \beta_{m'n'}}
    beta_11_beta_22 = 0;
    beta_11_beta_21 = 0;
    beta_12_beta_21 = 0;
    beta_12_beta_22 = 0;
    beta_21_beta_12 = 0;
    beta_21_beta_11 = 0; 
    beta_22_beta_11 = 0;
    beta_22_beta_12 = 0; 
    % Rows of the hessian 
    % note that as in gradient, alpha_21 is before alpha_12 (same for beta)
    hr1 = [nu_1_nu_1, nu_1_nu_2, nu_1_alpha_11, nu_1_alpha_21, nu_1_alpha_12, nu_1_alpha_22, nu_1_beta_11, nu_1_beta_21, nu_1_beta_12, nu_1_beta_22];
    hr2 = [nu_2_nu_1, nu_2_nu_2, nu_2_alpha_11, nu_2_alpha_21, nu_2_alpha_12, nu_2_alpha_22, nu_2_beta_11, nu_2_beta_21, nu_2_beta_12, nu_2_beta_22];
    hr3 = [alpha_11_nu_1, alpha_11_nu_2, alpha_11_alpha_11, alpha_11_alpha_21, alpha_11_alpha_12, alpha_11_alpha_22, alpha_11_beta_11, alpha_11_beta_21, alpha_11_beta_12, alpha_11_beta_22];
    hr4 = [alpha_21_nu_1, alpha_21_nu_2, alpha_21_alpha_11, alpha_21_alpha_21, alpha_21_alpha_12, alpha_21_alpha_22, alpha_21_beta_11, alpha_21_beta_21, alpha_21_beta_12, alpha_21_beta_22];
    hr5 = [alpha_12_nu_1, alpha_12_nu_2, alpha_12_alpha_11, alpha_12_alpha_21, alpha_12_alpha_12, alpha_12_alpha_22, alpha_12_beta_11, alpha_12_beta_21, alpha_12_beta_12, alpha_12_beta_22];
    hr6 = [alpha_22_nu_1, alpha_22_nu_2, alpha_22_alpha_11, alpha_22_alpha_12, alpha_22_alpha_21, alpha_22_alpha_22, alpha_22_beta_11, alpha_22_beta_21, alpha_22_beta_12, alpha_22_beta_22];      
    hr7 = [beta_11_nu_1, beta_11_nu_2, beta_11_alpha_11, beta_11_alpha_21, beta_11_alpha_12, beta_11_alpha_22, beta_11_beta_11, beta_11_beta_21, beta_11_beta_12, beta_11_beta_22];
    hr8 = [beta_21_nu_1, beta_21_nu_2, beta_21_alpha_11, beta_21_alpha_21, beta_21_alpha_12, beta_21_alpha_22, beta_21_beta_11, beta_21_beta_21, beta_21_beta_12, beta_21_beta_22];
    hr9 = [beta_12_nu_1, beta_12_nu_2, beta_12_alpha_11, beta_12_alpha_21, beta_12_alpha_12, beta_12_alpha_22, beta_12_beta_11, beta_12_beta_21, beta_12_beta_12, beta_12_beta_22];
    hr10 = [beta_22_nu_1, beta_22_nu_2, beta_22_alpha_11, beta_22_alpha_21, beta_22_alpha_12, beta_22_alpha_22, beta_22_beta_11, beta_22_beta_21, beta_22_beta_12, beta_22_beta_22];

    hessian = -[hr1;hr2;hr3;hr4;hr5;hr6;hr7;hr8;hr9;hr10];
    
elseif p==3
    % Trivariate case
    branch_ratio = a./b;
    times1 = times{1}'; times2 = times{2}'; times3 = times{3}';
    A_11 = zeros(length(times1),1); A_11(1) = 0;
    A_12 = zeros(length(times1),1); A_12(1) = 0;
    A_13 = zeros(length(times1),1); A_13(1) = 0;
    A_21 = zeros(length(times2),1); A_21(1) = 0;
    A_22 = zeros(length(times2),1); A_22(1) = 0;
    A_23 = zeros(length(times2),1); A_23(1) = 0;
    A_31 = zeros(length(times3),1); A_31(1) = 0;
    A_32 = zeros(length(times3),1); A_32(1) = 0;
    A_33 = zeros(length(times3),1); A_33(1) = 0;
    B_11 = 0; B_12 = 0; B_13 = 0; 
    B_21 = 0; B_22 = 0; B_23 = 0;  
    B_31 = 0; B_32 = 0; B_33 = 0; 
    % A_11 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tp_diff1 = diff(times1);
    for tp = 1:length(times1)-1
        A_11(tp+1) = (exp(-b(1,1)*tp_diff1(tp)))*(A_11(tp) + 1);
        B_11 = [B_11; sum((times1(tp+1)-times1(1:tp)).*exp(-b(1,1)*(times1(tp+1)-times1(1:tp))))];
    end
    % A_22 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tp_diff2 = diff(times2);
    for tp = 1:length(times2)-1
        A_22(tp+1) = (exp(-b(2,2)*tp_diff2(tp)))*(A_22(tp) + 1);
        B_22 = [B_22; sum((times2(tp+1)-times2(1:tp)).*exp(-b(2,2)*(times2(tp+1)-times2(1:tp))))];
    end
    % A_33 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tp_diff3 = diff(times3);
    for tp = 1:length(times3)-1
        A_33(tp+1) = (exp(-b(3,3)*tp_diff3(tp)))*(A_33(tp) + 1);
        B_33 = [B_33; sum((times3(tp+1)-times3(1:tp)).*exp(-b(3,3)*(times3(tp+1)-times3(1:tp))))];
    end
    % A_12 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp = 2:length(times1) 
        j_proc_times = intersect((times2(times2 < times1(tp))),(times2(times2 >= times1(tp-1))));
        A_12(tp) = (exp(-b(1,2)*tp_diff1(tp-1)))*A_12(tp-1) + sum(exp(-b(1,2)*(times1(tp) - j_proc_times)));
        times_set1 = [];
        times_set1 = times2(times2 < times1(tp));
        B_12 = [B_12; sum((times1(tp)-times_set1).*exp(-b(1,2)*(times1(tp)-times_set1)))];
    end
    % A_21 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp = 2:length(times2) 
        %j_proc_times = intersect((times1(times1 < times2(tp+1))),(times1(times1 > times2(tp))));
        j_proc_times = intersect((times1(times1 < times2(tp))),(times1(times1 >= times2(tp-1))));
        A_21(tp) = (exp(-b(2,1)*tp_diff2(tp-1)))*A_21(tp-1) + sum(exp(-b(2,1)*(times2(tp) - j_proc_times)));
        times_set2 = [];
        times_set2 = times1(times1 < times2(tp));
        B_21 = [B_21; sum((times2(tp)-times_set2).*exp(-b(2,1)*(times2(tp)-times_set2)))];
    end
    % A_13 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp = 2:length(times1) 
        j_proc_times = intersect((times3(times3 < times1(tp))),(times3(times3 >= times1(tp-1))));
        A_13(tp) = (exp(-b(1,3)*tp_diff1(tp-1)))*A_13(tp-1) + sum(exp(-b(1,3)*(times1(tp) - j_proc_times)));
        times_set3 = [];
        times_set3 = times3(times3 < times1(tp));
        B_13 = [B_13; sum((times1(tp)-times_set3).*exp(-b(1,3)*(times1(tp)-times_set3)))];
    end
    % A_31 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp = 2:length(times3) 
        j_proc_times = intersect((times1(times1 < times3(tp))),(times1(times1 >= times3(tp-1))));
        A_31(tp) = (exp(-b(3,1)*tp_diff3(tp-1)))*A_31(tp-1) + sum(exp(-b(3,1)*(times3(tp) - j_proc_times)));
        times_set4 = [];
        times_set4 = times1(times1 < times3(tp));
        B_31 = [B_31; sum((times3(tp)-times_set4).*exp(-b(3,1)*(times3(tp)-times_set4)))];
    end
    % A_23 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp = 2:length(times2) 
        j_proc_times = intersect((times3(times3 < times2(tp))),(times3(times3 >= times2(tp-1))));
        A_23(tp) = (exp(-b(2,3)*tp_diff2(tp-1)))*A_23(tp-1) + sum(exp(-b(2,3)*(times2(tp) - j_proc_times)));
        times_set5 = [];
        times_set5 = times3(times3 < times2(tp));
        B_23 = [B_23; sum((times2(tp)-times_set5).*exp(-b(2,3)*(times2(tp)-times_set5)))];
    end
    % A_32 terms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for tp = 2:length(times3)  
        j_proc_times = intersect((times2(times2 < times3(tp))),(times2(times2 >= times3(tp-1))));
        A_32(tp) = (exp(-b(3,2)*tp_diff3(tp-1)))*A_32(tp-1) + sum(exp(-b(3,2)*(times3(tp) - j_proc_times)));
        times_set6 = [];
        times_set6 = times2(times2 < times3(tp));
        B_32 = [B_32; sum((times3(tp)-times_set6).*exp(-b(3,2)*(times3(tp)-times_set6)))];
    end
    
    ll1_term1 = branch_ratio(1,1)*sum(exp(-b(1,1)*(end_time-times1))-1) + branch_ratio(1,2)*sum(exp(-b(1,2)*(end_time-times2))-1) + branch_ratio(1,3)*sum(exp(-b(1,3)*(end_time-times3))-1); 
    ll2_term1 = branch_ratio(2,2)*sum(exp(-b(2,2)*(end_time-times2))-1) + branch_ratio(2,1)*sum(exp(-b(2,1)*(end_time-times1))-1) + branch_ratio(2,3)*sum(exp(-b(2,3)*(end_time-times3))-1); 
    ll3_term1 = branch_ratio(3,3)*sum(exp(-b(3,3)*(end_time-times3))-1) + branch_ratio(3,1)*sum(exp(-b(3,1)*(end_time-times1))-1) + branch_ratio(3,2)*sum(exp(-b(3,2)*(end_time-times2))-1); 
    %    
    ll1_term2 = sum(log(v(1) + a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13)); 
    ll2_term2 = sum(log(v(2) + a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23)); 
    ll3_term2 = sum(log(v(3) + a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32)); 
    %
    log_ll1 = -v(1)*end_time + ll1_term1 + ll1_term2;
    log_ll2 = -v(2)*end_time + ll2_term1 + ll2_term2;
    log_ll3 = -v(3)*end_time + ll3_term1 + ll3_term2;
    
    log_ll = log_ll1 + log_ll2 + log_ll3;   
    
    % Gradient 
    log_ll = -log_ll; 
    exp_term11 = exp(-b(1,1)*(end_time-times1)); 
    exp_term12 = exp(-b(1,2)*(end_time-times2)); 
    exp_term13 = exp(-b(1,3)*(end_time-times3)); 
    exp_term21 = exp(-b(2,1)*(end_time-times1)); 
    exp_term22 = exp(-b(2,2)*(end_time-times2)); 
    exp_term23 = exp(-b(2,3)*(end_time-times3)); 
    exp_term31 = exp(-b(3,1)*(end_time-times1)); 
    exp_term32 = exp(-b(3,2)*(end_time-times2)); 
    exp_term33 = exp(-b(3,3)*(end_time-times3)); 
    
    gradient_baseline1 = -end_time + sum(1./(v(1)+a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13));  
    gradient_baseline2 = -end_time + sum(1./(v(2)+a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23));  
    gradient_baseline3 = -end_time + sum(1./(v(3)+a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32));  
    grads_base = [gradient_baseline1;gradient_baseline2;gradient_baseline3];

    gradient_A11 = sum((1/b(1,1))*(exp_term11-1)) + sum((A_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13));
    gradient_A12 = sum((1/b(1,2))*(exp_term12-1)) + sum((A_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13));
    gradient_A13 = sum((1/b(1,3))*(exp_term13-1)) + sum((A_13)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13));
    gradient_A21 = sum((1/b(2,1))*(exp_term21-1)) + sum((A_21)./(v(2)+a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23));  
    gradient_A22 = sum((1/b(2,2))*(exp_term22-1)) + sum((A_22)./(v(2)+a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23));  
    gradient_A23 = sum((1/b(2,3))*(exp_term23-1)) + sum((A_23)./(v(2)+a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23));
    gradient_A31 = sum((1/b(3,1))*(exp_term31-1)) + sum((A_31)./(v(3)+a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32));
    gradient_A32 = sum((1/b(3,2))*(exp_term32-1)) + sum((A_32)./(v(3)+a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32));
    gradient_A33 = sum((1/b(3,3))*(exp_term33-1)) + sum((A_33)./(v(3)+a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32));
    gradsA = [gradient_A11;gradient_A21;gradient_A31;gradient_A12;gradient_A22;gradient_A32;gradient_A13;gradient_A23;gradient_A33];

    gradient_B11 = -a(1,1)*(sum( (1/b(1,1))*(end_time-times1).*exp_term11 + (1/(b(1,1)^2))*(exp_term11-1))) - sum((a(1,1)*B_11)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13));
    gradient_B12 = -a(1,2)*(sum( (1/b(1,2))*(end_time-times2).*exp_term12 + (1/(b(1,2)^2))*(exp_term12-1))) - sum((a(1,2)*B_12)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13));
    gradient_B13 = -a(1,3)*(sum( (1/b(1,3))*(end_time-times3).*exp_term13 + (1/(b(1,3)^2))*(exp_term13-1))) - sum((a(1,3)*B_13)./(v(1)+a(1,1)*A_11 + a(1,2)*A_12 + a(1,3)*A_13));
    gradient_B21 = -a(2,1)*(sum( (1/b(2,1))*(end_time-times1).*exp_term21 + (1/(b(2,1)^2))*(exp_term21-1))) - sum((a(2,1)*B_21)./(v(2)+a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23));  
    gradient_B22 = -a(2,2)*(sum( (1/b(2,2))*(end_time-times2).*exp_term22 + (1/(b(2,2)^2))*(exp_term22-1))) - sum((a(2,2)*B_22)./(v(2)+a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23));
    gradient_B23 = -a(2,3)*(sum( (1/b(2,3))*(end_time-times3).*exp_term23 + (1/(b(2,3)^2))*(exp_term23-1))) - sum((a(2,3)*B_23)./(v(2)+a(2,2)*A_22 + a(2,1)*A_21 + a(2,3)*A_23));
    gradient_B31 = -a(3,1)*(sum( (1/b(3,1))*(end_time-times1).*exp_term31 + (1/(b(3,1)^2))*(exp_term31-1))) - sum((a(3,1)*B_31)./(v(3)+a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32));
    gradient_B32 = -a(3,2)*(sum( (1/b(3,2))*(end_time-times2).*exp_term32 + (1/(b(3,2)^2))*(exp_term32-1))) - sum((a(3,2)*B_32)./(v(3)+a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32));
    gradient_B33 = -a(3,3)*(sum( (1/b(3,3))*(end_time-times3).*exp_term33 + (1/(b(3,3)^2))*(exp_term33-1))) - sum((a(3,3)*B_33)./(v(3)+a(3,3)*A_33 + a(3,1)*A_31 + a(3,2)*A_32));
    gradsB = [gradient_B11;gradient_B21;gradient_B31;gradient_B12;gradient_B22;gradient_B32;gradient_B13;gradient_B23;gradient_B33];
    grad = -[grads_base; gradsA; gradsB];
    
    

end

end
