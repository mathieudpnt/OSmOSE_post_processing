function [overlap_rate_result] = overlap_rate(vect1,vect2)

[~, interval_0] = intersection_vect(vect1, vect2);

for i = 1:length(interval_0) %vect1
    interval = interval_0{i};
    
    
    for j = 1:height(interval) %vect2
        if isdatetime(interval)
            condition = sum(isnat( interval(j,:) ));
        elseif isnumeric(interval)
            condition = sum(isnan( interval(j,:) ));
        else
            disp('error...')
            return
        end
        
        if condition == 0
            recouvrement = interval(j,2) - interval(j,1);
            longueur_ref = max((vect1(i,2) - vect1(i,1)), (vect2(j,2) - vect2(j,1)));
            overlap_rate_result(j,i) = recouvrement/longueur_ref;
        else
            overlap_rate_result(j,i) = 0;
        end
    end
    
end

end

%%
% function [overlap_rate_result] = overlap_rate(vect1,vect2)
% 
% [~, interval] = intersection_vect(vect1, vect2);
% 
% 
% 
% if ~isempty(interval)
%     recouvrement = interval(2) - interval(1);
%     longueur_ref = max((vect1(2) - vect1(1)), (vect2(2) - vect2(1)));
%     overlap_rate_result = recouvrement/longueur_ref;
% else
%     overlap_rate_result = 0;
% end
% 
% end