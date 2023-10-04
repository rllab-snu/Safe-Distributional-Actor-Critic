#include <iostream>

using namespace std;

extern "C" {
    void projection(int n_quantiles1, double weight1, double* quantiles1, int n_quantiles2, double weight2, double* quantiles2, int n_quantiles3, double* new_quantiles){
        /* 
            different numbers of quanitles.
            should be "n_quantiles1 <= n_quantiles2".
            interpolation: linear 
        */
        int current_idx;
        int total_quantiles_num = n_quantiles1 + n_quantiles2;
        double* integ_pmf = new double[total_quantiles_num];
        double* integ_cmf = new double[total_quantiles_num];
        double* sorted_quantiles = new double[total_quantiles_num];
        double curr_cum_prob, next_cum_prob, target_cum_prob, curr_quantile, next_quantile;

        int idx1 = 0, idx2 = 0;
        for(int i=0; i<total_quantiles_num; i++){
            if(idx2 == n_quantiles2 || (idx1 != n_quantiles1 && quantiles1[idx1] < quantiles2[idx2])){
                integ_pmf[i] = weight1/(n_quantiles1*(weight1 + weight2));
                sorted_quantiles[i] = quantiles1[idx1];
                idx1++;
            }else{
                integ_pmf[i] = weight2/(n_quantiles2*(weight1 + weight2));
                sorted_quantiles[i] = quantiles2[idx2];
                idx2++;
            }
            if(i == 0) integ_cmf[i] = integ_pmf[i];
            else integ_cmf[i] = integ_cmf[i - 1] + integ_pmf[i];
        }

        current_idx = 0;
        for(int i=0; i<total_quantiles_num; i++){
            if(i == 0){
                curr_quantile = sorted_quantiles[0];
                curr_cum_prob = 0.0;
            }else{
                curr_quantile = sorted_quantiles[i-1];
                curr_cum_prob = integ_cmf[i-1] - integ_pmf[i-1]/2.0;
            }
            next_quantile = sorted_quantiles[i];
            next_cum_prob = integ_cmf[i] - integ_pmf[i]/2.0;
            while(true){
                target_cum_prob = (current_idx + 0.5)/n_quantiles3;
                if(current_idx < n_quantiles3 && target_cum_prob >= curr_cum_prob && target_cum_prob < next_cum_prob){
                    new_quantiles[current_idx] = ((next_cum_prob - target_cum_prob)*curr_quantile + (target_cum_prob - curr_cum_prob)*next_quantile)/(next_cum_prob - curr_cum_prob);
                    current_idx++;
                }else{
                    break;
                }
            }
            if(current_idx == n_quantiles3) break;
        }
        while(current_idx < n_quantiles3){
            new_quantiles[current_idx] = sorted_quantiles[total_quantiles_num-1];
            current_idx++;
        }

        delete [] integ_pmf;
        delete [] integ_cmf;
        delete [] sorted_quantiles;
    }
}