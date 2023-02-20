function [intersection, interval] = intersection_vect(vect1, vect2)

% Given:
% 2 Intervals
% Interval 1: (start1, end1)
% Interval 2: (start2, end2)
% 
% Following TWO conditions should be true to consider that 2 intervals are intersected:
% end2 >= start1 AND start2 <= end1
% OR
% end1 >= start2 AND start1 <= end2

if (vect2(2) >= vect1(1) && vect2(1) <= vect1(2) ) || (vect1(2) >= vect2(1) && vect1(1) <= vect2(2) )
    intersection = 1;
    
    if vect1(1)<=vect2(1) && vect1(2)<=vect2(2)
        if abs(vect2(1)- vect1(2)) < 1 % less than 1sec
            interval = [];  
        else
            interval = [vect2(1) vect1(2)];
        end
    elseif vect2(1)<=vect1(1) && vect2(2)<=vect1(2)
        if abs(vect1(1)- vect2(2))  < 1 % less than 1sec
            interval = [];  
        else
            interval = [vect1(1) vect2(2)];
        end
    elseif vect1(1)>=vect2(1) && vect1(2)<=vect2(2)
        interval = vect1;
    elseif vect2(1)>=vect1(1) && vect2(2)<=vect1(2)
        interval = vect2;
    elseif vect1(1)>=vect2(1) && vect1(2)>=vect2(2)
        if abs(vect1(1)- vect2(2)) < 1
            interval = [vect1(1) vect2(2)];   
        else
            interval = [];  
        end
    elseif vect2(1)>=vect1(1) && vect2(2)>=vect1(2)
        if abs(vect2(1)- vect1(2)) < 1
            interval = [];  
        else
            interval = [vect2(1) vect1(2)]; 
        end
    end
    

else
    intersection = 0;
    interval = [];
end
    
end
