Bias Corrected and Accelerated Bootstrap Confidence Intervals in Python

The bootstrap is commonly used by analysts as an intuitive, nonparametric method to get confidence intervals. Like any tool that is so powerful, it is worth understanding why it works.

https://projecteuclid.org/euclid.aos/1176344552

https://projecteuclid.org/download/pdf_1/euclid.aos/1176345338

https://statweb.stanford.edu/~ckirby/brad/papers/2018Automatic-Construction-BCIs.pdf
https://faculty.washington.edu/heagerty/Courses/b572/public/GregImholte-1.pdf
https://www.tau.ac.il/~saharon/Boot/10.1.1.133.8405.pdf
http://sumsar.net/blog/2015/04/the-non-parametric-bootstrap-as-a-bayesian-model/

https://stats.stackexchange.com/questions/129478/when-is-the-bootstrap-estimate-of-bias-valid
http://faculty.washington.edu/fscholz/Reports/InfinitesimalJackknife.pdf

https://www.stat.cmu.edu/~cshalizi/uADA/13/lectures/which-bootstrap-when.pdf

https://stats.stackexchange.com/questions/26088/explaining-to-laypeople-why-bootstrapping-works/26093

A whole book 
http://www.ru.ac.bd/stat/wp-content/uploads/sites/25/2019/03/501_02_Efron_Introduction-to-the-Bootstrap.pdf

Should I just always use the bootstrap?

It may need to be adapted if there's dependent data (block bootstrap)

It may act weird if the  statistic can change a ton if you change one data point

Sometimes the standard intervals are great and fast

It makes weird model assumptions

Serious data analyses should always include serious consideration of model constraints; although knowledge of the context of a data set may make the incorporation of reasonable model constraints obvious, and although the bootstrap and the Bayesian bootstrap may be useful in many particular contexts, there are no general data analytic panaceas that allow us to pull ourselves up by our bootstraps.
