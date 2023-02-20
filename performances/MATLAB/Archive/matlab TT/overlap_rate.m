function [overlap_rate] = overlap_rate(vect1,vect2)

[~, interval] = intersection_vect(vect1, vect2);

if any(interval)
    recouvrement = interval(2) - interval(1);
    longueur_ref = max((vect1(2) - vect1(1)), (vect2(2) - vect2(1)));
    overlap_rate = recouvrement/longueur_ref;
else
    overlap_rate = 0;
end

end