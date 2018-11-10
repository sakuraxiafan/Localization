function y = my_cut(x,rate1,rate2)
y = x;
v = sort(x(:));
m = [v(max(1,round(rate1*length(v)))) v(round(rate2*length(v)))];
y(y<m(1))=m(1); y(y>m(2))=m(2);
end