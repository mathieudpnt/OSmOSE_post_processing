function [result_intersection, result_interval] = intersection_vect(vect1, vect2)

% Given:
% 2 Intervals
% Interval 1: (start1, end1)
% Interval 2: (start2, end2)
% 
% Following TWO conditions should be true to consider that 2 intervals are intersected:
% end2 >= start1 AND start2 <= end1
% OR
% end1 >= start2 AND start1 <= end2

if isnumeric(vect1) && isnumeric(vect2)
    condition_num = 1;
    condition_datetime = 0;
elseif isdatetime(vect1) && isdatetime(vect2)
    condition_num = 0;
    condition_datetime = 1;
else
    disp('error data type')
    return
end


for k = 1:height(vect1)
    for i = 1:height(vect2)
        if (vect2(i,2) >= vect1(k,1) && vect2(i,1) <= vect1(k,2) ) || (vect1(k,2) >= vect2(i,1) && vect1(k,1) <= vect2(i,2) )
            intersection(i,1) = 1;

            if vect1(k,1)<=vect2(i,1) && vect1(k,2)<=vect2(i,2)
                if abs(vect2(i,1)- vect1(k,2)) < seconds(1) % less than 1sec
    %                 interval(i,:) = []; 
                    if condition_num == 1
                        interval(i,:) = [NaN NaN];
                    elseif condition_datetime == 1
                        interval(i,:) = [NaT NaT];
                    else
                        disp('error data type 1')
                        return
                    end
                else
                    interval(i,:) = [vect2(i,1) vect1(k,2)];
                end
            elseif vect2(i,1)<=vect1(k,1) && vect2(i,2)<=vect1(k,2)
                if abs(vect1(k,1)- vect2(i,2))  < seconds(1) % less than 1sec
    %                 interval(i,:) = [];  
                    if condition_num == 1
                        interval(i,:) = [NaN NaN];
                    elseif condition_datetime == 1
                        interval(i,:) = [NaT NaT];
                    else
                        disp('error data type 2')
                        return
                    end
                else
                    interval(i,:) = [vect1(k,1) vect2(i,2)];
                end
            elseif vect1(k,1)>=vect2(i,1) && vect1(k,2)<=vect2(i,2)
                interval(i,:) = vect1(k,:);
            elseif vect2(i,1)>=vect1(k,1) && vect2(i,2)<=vect1(k,2)
                interval(i,:) = vect2(i,:);
            elseif vect1(k,1)>=vect2(i,1) && vect1(k,2)>=vect2(i,2)
                if abs(vect1(k,1)- vect2(i,2)) < 1
                    interval(i,:) = [vect1(k,1) vect2(i,2)];   
                else
    %                 interval(i,:) = [];  
                    if condition_num == 1
                        interval(i,:) = [NaN NaN];
                    elseif condition_datetime == 1
                        interval(i,:) = [NaT NaT];
                    else
                        disp('error data type 3')
                        return
                    end
                end
            elseif vect2(i,1)>=vect1(k,1) && vect2(i,2)>=vect1(k,2)
                if abs(vect2(i,1)- vect1(k,2)) < 1
    %                 interval(i,:) = [];  
                    if condition_num == 1
                        interval(i,:) = [NaN NaN];
                    elseif condition_datetime == 1
                        interval(i,:) = [NaT NaT];
                    else
                        disp('error data type 4')
                        return
                    end
                else
                    interval(i,:) = [vect2(i,1) vect1(k,2)]; 
                end
            end

        else
            intersection(i,1) = 0;
    %         interval(i,:) = [];
                    if condition_num == 1
                        interval(i,:) = [NaN NaN];
                    elseif condition_datetime == 1
                        interval(i,:) = [NaT NaT];
                    else
                        disp('error data type 5')
                        return
                    end
        end
    result_intersection{k} = intersection;
    result_interval{k} = interval;
    end
end

end
%%
% function [intersection, interval] = intersection_vect(vect1, vect2)
% 
% % Given:
% % 2 Intervals
% % Interval 1: (start1, end1)
% % Interval 2: (start2, end2)
% % 
% % Following TWO conditions should be true to consider that 2 intervals are intersected:
% % end2 >= start1 AND start2 <= end1
% % OR
% % end1 >= start2 AND start1 <= end2
% 
% if (vect2(2) >= vect1(1) && vect2(1) <= vect1(2) ) || (vect1(2) >= vect2(1) && vect1(1) <= vect2(2) )
%     intersection = 1;
%     
%     if vect1(1)<=vect2(1) && vect1(2)<=vect2(2)
%         if abs(vect2(1)- vect1(2)) < seconds(1) % less than 1sec
%             interval = [];  
%         else
%             interval = [vect2(1) vect1(2)];
%         end
%     elseif vect2(1)<=vect1(1) && vect2(2)<=vect1(2)
%         if abs(vect1(1)- vect2(2))  < seconds(1) % less than 1sec
%             interval = [];  
%         else
%             interval = [vect1(1) vect2(2)];
%         end
%     elseif vect1(1)>=vect2(1) && vect1(2)<=vect2(2)
%         interval = vect1;
%     elseif vect2(1)>=vect1(1) && vect2(2)<=vect1(2)
%         interval = vect2;
%     elseif vect1(1)>=vect2(1) && vect1(2)>=vect2(2)
%         if abs(vect1(1)- vect2(2)) < 1
%             interval = [vect1(1) vect2(2)];   
%         else
%             interval = [];  
%         end
%     elseif vect2(1)>=vect1(1) && vect2(2)>=vect1(2)
%         if abs(vect2(1)- vect1(2)) < 1
%             interval = [];  
%         else
%             interval = [vect2(1) vect1(2)]; 
%         end
%     end
%     
% 
% else
%     intersection = 0;
%     interval = [];
% end
%     
% end
