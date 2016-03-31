function M = LoGMatrix(w, sigma)
M = zeros(w*2+1, w*2+1);
for y=-w:w
    yy = y+w+1;
    for x=-w:w
        xx = x+w+1;
        val = exp(-(x*x+y*y)/(2*sigma*sigma));
        M(yy, xx) = val;
    end
end

M = M ./ sum(sum(M));

for y=-w:w
    yy = y+w+1;
    for x=-w:w
        xx = x+w+1;   
        M(yy, xx) = M(yy, xx) * (x*x+y*y - 2*sigma*sigma);
    end
end
M = M - sum(sum(M))/(size(M, 1)*size(M,2));
end