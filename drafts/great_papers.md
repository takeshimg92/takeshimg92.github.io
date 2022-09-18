n non-technical papers that made me a better Data Scientist

# [Statistical Modeling: The Two Cultures - Leo Breiman](https://projecteuclid.org/euclid.ss/1009213726)

- Breiman's view is as someone who left the statistical academic mainstream to work in industry, and then returned to Berkeley's statistics department confused by the disconnect between statistical theory and data analytic practice. 

- Particularly, his critique was that [??? wow it's hard to say this concisely]

- This 

- Breiman's critique of statistics is now about twenty years old, and it is clear that his belief that algorithmic modeling would come to dominate industry data analysis was very clearly on the mark. And while I'm not an academic statistician, his description of "two cultures" resonates with my experience. Data analysis is a field which draws a wide variety of experienced professionals which an equally varied set of perspectives.

- In particular, I've noticed a particular kind of professional misunderstanding that can happen between two people with the title "data scientist" - predictive model person vs causal inference person

- For an IC, it makes sense to think about what culture you have more experience with - often, this will highlight blind spots in your own thinking, and give you ways to connect with your colleagues' ideas

# [Statistical Inference: The Big Picture - Robert E. Kass](http://www.stat.cmu.edu/~kass/papers/bigpic.pdf)

An interesting response to this is [Bayesian Statistical Pragmatism - Andrew Gelman](https://projecteuclid.org/euclid.ss/1307626555)

# [Statistical Models and Shoe Leather - David A. Freedman](https://psychology.okstate.edu/faculty/jgrice/psyc5314/Freedman_1991A.pdf)

Is regression actually a causal inference success story?

According to Freedman, the results of the adoption of regression have been unimpressive. Practitioners without great statistical depth (like John Snow) can achieve good results by careful causal reasoning, combining many lines of evidence, and by putting in the effort to understand how data was collected. Those who attempt causal inference from regression should take note of the rigor and success of such examples despite their lack of technical sophistication.

> What is the difference between Kanarek et al.'s study and Snow's? Kanarek et al. ignored the ecological fallacy. Snow dealt with it. Kanarek et al. tried to control for covariates by modeling, using socioeconomic status as a proxy for smoking. Snow found a natural experiment and collected the data he needed. Kanarek et al.'s argument for causation rides on the statistical significance of a coefficient. Snow's argument used logic and shoe leather. Regression models make it all too easy to substitute technique for work.

A deep knowledge of the data generating process is required in order to make causal claims - Snow put in the Shoe Leather to get that knowledge by collecting the data himself, but often causality-from-regression practitioners expect statistical wizardry to make this work unnecessary

# [To Explain or Predict? - Galit Shmueli](https://www.stat.berkeley.edu/~aldous/157/Papers/shmueli.pdf)

# [What You Can and Can’t Properly Do with Regression - Richard Berk](http://www.public.asu.edu/~gasweete/crj604/readings/2010-Berk%20(what%20you%20can%20and%20can't%20do%20with%20regression).pdf)

# [Philosophy and the practice of Bayesian Statistics - Andrew Gelman, Cosma Rohilla Shalizi](http://www.stat.columbia.edu/~gelman/research/published/philosophy.pdf)

# [Confessions of a Pragmatic Statistician - Chris Chatfield](https://www2.isye.gatech.edu/isyebayes/bank/chatfield.pdf)

# Some themes

Be mindful of what your P-values are actually telling you; we often assume a significant result is a confirmation of our hypothesis but it is usually much more limited than that

You need to understand how the data was generated if you want to make causal claims
