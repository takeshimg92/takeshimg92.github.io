---
layout: post
title:  "Some tests"
date:   2022-09-18 00:00:00
categories: data science machine learning
tags: featured
image: /assets/article_images/2014-08-29-welcome-to-jekyll/desktop.JPG
mathjax: true
---

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.


```python3
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probs)
# >> 0.867
```

```python
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probs)
# >> 0.867
```

```{python}
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probs)
# >> 0.867
```

Let us try out some equations!

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^ 4} T_{\mu \nu}$$

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll’s dedicated Help repository][jekyll-help].

{% highlight js %}

<footer class="site-footer">
 <a class="subscribe" href="{{ "/feed.xml" | prepend: site.baseurl }}"> <span class="tooltip"> <i class="fa fa-rss"></i> Subscribe!</span></a>
  <div class="inner">a
   <section class="copyright">All content copyright <a href="mailto:{{ site.email}}">{{ site.name }}</a> &copy; {{ site.time | date: '%Y' }} &bull; All rights reserved.</section>
   <section class="poweredby">Made with <a href="http://jekyllrb.com"> Jekyll</a></section>
  </div>
</footer>
{% endhighlight %}


[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
