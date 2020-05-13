<body>
<div class="document" id="a-simple-recommender-system-with-observed-attributes-and-time-varying-taste-parameters">
<h1 class="title">A Simple Recommender System with Observed Attributes and Time-Varying Taste Parameters</h1>

<p>This document provides a brief explanation of the <tt class="docutils literal">recommendx</tt> package for Python. This package implements a recommender
system, similar to the matrix factorization-based algorithms (SVD) available in the <strong>excellent</strong>
<a class="reference external" href="http://surpriselib.com/">Surprise</a> package.</p>
<p>This package extends the standard SVD recommender system by allowing researchers to include observed item attributes and
also user taste parameters that vary over time. The model is fit using stochastic gradient descent.</p>
<p>If you use this package, please cite the following working paper:</p>
<ul class="simple">
<li>Adam D. Rennhoff (2020): &quot;A Simple Recommender System with Observed Attributes and Time-Varying Parameters,&quot;MTSU Working Paper #XXXX (TO BE UPDATED WITH NUMBER AND LINK)</li>
</ul>
<p>The paper contains a number of helpful examples and suggestions for implementation.
Please consult the paper for help on using <tt class="docutils literal">recommendx</tt> for your research.</p>
<p>This package is distributed under the <a class="reference external" href="https://opensource.org/licenses/BSD-3-Clause">BSD 3-Clause license</a>.</p>
<div class="section" id="installation">
<h1>Installation</h1>
<p><tt class="docutils literal">recommendx</tt> has the following dependencies:</p>
<ul class="simple">
<li>Python (&gt;=3.5)</li>
<li>NumPy (&gt;= 1.10)</li>
<li>Pandas (&gt;= 0.18)</li>
</ul>
<p>The easiest way to install <tt class="docutils literal">recommendx</tt> is using <tt class="docutils literal">pip</tt>:</p>
<pre class="literal-block">
pip install recommendx
</pre>
<p>Alternatively, the package can be accessed from Github.</p>
<p>The package contains two related prediction algorithms: <tt class="docutils literal">RWR</tt> and <tt class="docutils literal">RWT</tt>. These are discussed below.</p>
</div>
<div class="section" id="recommendation-with-regressors-rwr">
<h1>Recommendation with Regressors (RWR)</h1>
<p><tt class="docutils literal">RWR</tt> implements a slightly modified version of what might we might call the &quot;classic&quot; SVD algorithm. This is often
attributed to <a class="reference external" href="https://sifter.org/~simon/journal/20061211.html">Simon Funk</a>, who famously used it during the
<a class="reference external" href="https://www.netflixprize.com/">Netflix Prize</a> competition. The classic SVD approach relies only upon latent item
attributes. <tt class="docutils literal">RWR</tt> extends this framework by allowing the researcher to specify observed item attributes, as well.</p>
<p>We can define <span class="formula"><i>r̂</i><sub><i>ui</i></sub></span> as user <span class="formula"><i>u</i></span>'s predicted rating for item <span class="formula"><i>i</i></span>:</p>
<div class="formula">
<i>r̂</i><sub><i>ui</i></sub> = <i>μ</i> + <i>b</i><sub><i>u</i></sub> + <i>X</i><sub><i>i</i></sub><i>β</i><sub><i>u</i></sub> + <i>Z</i><sub><i>i</i></sub><i>α</i><sub><i>u</i></sub>
</div>
<p>In this specification,</p>
<ul class="simple">
<li><span class="formula"><i>μ</i></span> is the average rating in the data</li>
<li><span class="formula"><i>b</i><sub><i>u</i></sub></span> is the bias for user <span class="formula"><i>u</i></span></li>
<li><span class="formula"><i>X</i><sub><i>i</i></sub></span> is a vector of <strong>observed</strong> attributes for item <span class="formula"><i>i</i></span></li>
<li><span class="formula"><i>Z</i><sub><i>i</i></sub></span> is a vector of <strong>latent</strong> attributes for item <span class="formula"><i>i</i></span></li>
<li><span class="formula"><i>β</i><sub><i>u</i></sub></span> and <span class="formula"><i>α</i><sub><i>u</i></sub></span> are user <span class="formula"><i>u</i></span>'s preferences for observed and latent item attributes, respectively</li>
</ul>
<p>This specification is similar to the usual matrix factorization set-up, with the standard item bias term (<span class="formula"><i>b</i><sub><i>i</i></sub></span>)
replaced by observed attributes.</p>
<p>Defining <span class="formula"><i>R</i></span> as the set of all observed user-item ratings
and imposing L2-regularization on our parameters, we seek to minimize the following objective function:</p>
<div class="formula">
<span class="limits"><sup class="limit"> </sup><span class="limit">Σ</span><sub class="limit"><i>r</i><sub><i>ui</i></sub> ∈ <i>R</i></sub></span> = (<i>r</i><sub><i>ui</i></sub> − <i>r̂</i><sub><i>ui</i></sub>)<sup>2</sup> + <i>λ</i>(<i>b</i><span class="scripts"><sup class="script">2</sup><sub class="script"><i>u</i></sub></span> + ∣∣<i>β</i><sub><i>u</i></sub>∣∣<sup>2</sup> + ∣∣<i>Z</i><sub><i>i</i></sub>∣∣<sup>2</sup> + ∣∣<i>α</i><sub><i>u</i></sub>∣∣<sup>2</sup>)
</div>
<p>The minimization is done using stochastic gradient descent (SGD).
The relevant gradients, which can easily be obtained by hand, lead to the following update rules:</p>
<ul class="simple">
<li><span class="formula"><i>b</i><sub><i>u</i></sub><span class="text"> </span>⟸<span class="text"> </span><i>b</i><sub><i>u</i></sub> + <i>γ</i>(<i>e</i><sub><i>ui</i></sub> − <i>λ</i><i>b</i><sub><i>u</i></sub>)</span></li>
<li><span class="formula"><i>β</i><sub><i>u</i></sub><span class="text"> </span>⟸<span class="text"> </span><i>β</i><sub><i>u</i></sub> + <i>γ</i>(<i>e</i><sub><i>ui</i></sub><i>X</i><sub><i>i</i></sub> − <i>λ</i><i>β</i><sub><i>u</i></sub>)</span></li>
<li><span class="formula"><i>α</i><sub><i>u</i></sub><span class="text"> </span>⟸<span class="text"> </span><i>α</i><sub><i>u</i></sub> + <i>γ</i>(<i>e</i><sub><i>ui</i></sub><i>Z</i><sub><i>i</i></sub> − <i>λ</i><i>α</i><sub><i>u</i></sub>)</span></li>
<li><span class="formula"><i>Z</i><sub><i>i</i></sub><span class="text"> </span>⟸<span class="text"> </span><i>Z</i><sub><i>i</i></sub> + <i>γ</i>(<i>e</i><sub><i>ui</i></sub><i>α</i><sub><i>i</i></sub> − <i>λ</i><i>Z</i><sub><i>i</i></sub>)</span></li>
</ul>
<p>where <span class="formula"><i>e</i><sub><i>ui</i></sub> = <i>r</i><sub><i>ui</i></sub> − <i>r̂</i><sub><i>ui</i></sub></span>, <span class="formula"><span class="text"> </span><i>λ</i></span> is the regularization penalty term, and
<span class="formula"><i>γ</i></span> is the learning rate. The learning rates determines how large of a &quot;step&quot; to take when we update
parameters.</p>
<div class="section" id="parameters">
<h2>Parameters</h2>
<p>Note: I have purposely chosen the parameter names to be similar to those in <a class="reference external" href="http://surpriselib.com/">Surprise</a>
in order to facilitate easy movement between packages.</p>
<ul class="simple">
<li><strong>n_factors</strong> - The number of latent factors in <span class="formula"><i>Z</i></span>. Default is <tt class="docutils literal">50</tt>.</li>
<li><strong>n_epochs</strong> - The number of iterations of the SGD procedure. Default is <tt class="docutils literal">50</tt>.</li>
<li><strong>init_mean</strong> - The mean of the normal distribution used to initialize parameter values. Default is <tt class="docutils literal">0</tt>.</li>
<li><strong>init_std_dev</strong> - The standard deviation of the normal distribution used to initialize parameter values. Default is <tt class="docutils literal">0.1</tt>.</li>
<li><strong>reg</strong> - The regularization term used for all parameters (<span class="formula"><i>λ</i></span>). Default is <tt class="docutils literal">0.02</tt>.</li>
<li><strong>lr</strong> - The learning rate for all parameters (<span class="formula"><i>γ</i></span>). Default is <tt class="docutils literal">0.005</tt>.</li>
</ul>
</div>
<div class="section" id="attributes">
<h2>Attributes</h2>
<p>Once an <tt class="docutils literal">RWR</tt> instance is <tt class="docutils literal">fit()</tt>, the resulting parameter values are returned
as attributes of the instance.</p>
<ul class="simple">
<li><dl class="first docutils">
<dt><tt class="docutils literal">intercept_</tt>: (<span class="formula"><i>μ</i></span>)</dt>
<dd>Scalar intercept term</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">bu</tt>: (<span class="formula"><i>b</i><sub><i>u</i></sub></span>)</dt>
<dd>NumPy array with shape (n_users, 1)</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">B</tt>: (<span class="formula"><i>β</i><sub><i>u</i></sub></span>)</dt>
<dd>If <span class="formula"><i>X</i><sub><i>i</i></sub></span> is provided to <tt class="docutils literal">fit()</tt> (see below), <tt class="docutils literal">B</tt> is a NumPy array with shape (n_users, n_Xs)</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">alpha_</tt>: (<span class="formula"><i>α</i><sub><i>u</i></sub></span>)</dt>
<dd>NumPy array with shape (n_users, n_factors)</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">Z</tt>: (<span class="formula"><i>Z</i><sub><i>i</i></sub></span>)</dt>
<dd>NumPy array with shape (u_items, n_factors)</dd>
</dl>
</li>
</ul>
</div>
<div class="section" id="methods">
<h2>Methods</h2>
<ul class="simple">
<li><dl class="first docutils">
<dt><tt class="docutils literal">fit(self,df,Xi=None)</tt></dt>
<dd><ul class="first last">
<li>Fits the recommender system model</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">df</tt> must be a NumPy array</dt>
<dd><ul class="first last">
<li>Each row corresponds to a rating (<span class="formula"><i>r</i><sub><i>ui</i></sub></span>)</li>
<li>Columns <strong>must</strong> be ordered: [user, item, rating]</li>
<li><tt class="docutils literal">user</tt> and <tt class="docutils literal">item</tt> may be strings or integers</li>
</ul>
</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">Xi</tt> (if supplied) must be a NumPy array</dt>
<dd><ul class="first last">
<li>If no observed item attributes are supplied, <tt class="docutils literal">fit()</tt> returns the same results as SVD</li>
<li>First column of <tt class="docutils literal">Xi</tt> must be item identifier that corresponds with item labels used in <tt class="docutils literal">df</tt></li>
<li>Shape of array is (n_items, 1 + n_Xs)</li>
</ul>
</dd>
</dl>
</li>
</ul>
</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">accuracy(self,df,Xi=None)</tt></dt>
<dd><ul class="first last">
<li>Returns the mean squared prediction error</li>
<li>Requires the recommender system to be fit first</li>
<li>All provided values must be in the same format as supplied to <tt class="docutils literal">fit()</tt></li>
</ul>
</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">predict(self,u_p,i_p)</tt></dt>
<dd><ul class="first last">
<li>Returns predicted ratings</li>
<li><tt class="docutils literal">u_p</tt> is a user value</li>
<li><tt class="docutils literal">i_p</tt> is an item value</li>
<li>Both <tt class="docutils literal">u_p</tt> and <tt class="docutils literal">i_p</tt> must be provided in the same format as <tt class="docutils literal">fit()</tt></li>
</ul>
</dd>
</dl>
</li>
</ul>
</div>
<div class="section" id="sample-syntax">
<h2>Sample Syntax</h2>
<p>If we assume that <tt class="docutils literal">dat</tt> is a NumPy array of ratings data and <tt class="docutils literal">att</tt> is a NumPy array
of observed item attributes, we can use the following code:</p>
<pre class="literal-block">
from recommendx import RWR
rwr = RWR(n_factors = 5)
rwr.fit(dat,att)
rwr.accuracy(dat,att)
rwr.predict('userA','item10')
</pre>
</div>
</div>
<div class="section" id="recommendation-with-time-rwt">
<h1>Recommendation with Time (RWT)</h1>
<p><tt class="docutils literal">RWT</tt> implements the same basic model as <tt class="docutils literal">RWR</tt> but allows for time-varying taste parameters.</p>
<p>Our main ratings prediction equation becomes:</p>
<div class="formula">
<i>r̂</i><sub><i>uit</i></sub> = <i>μ</i> + <i>b</i><sub><i>u</i></sub> + <i>X</i><sub><i>i</i></sub><i>β</i><sub><i>u</i>, <i>t</i></sub> + <i>Z</i><sub><i>i</i></sub><i>α</i><sub><i>u</i>, <i>t</i></sub>
</div>
<p>Neither observed (<span class="formula"><i>X</i><sub><i>i</i></sub></span>) nor unobserved (<span class="formula"><i>Z</i><sub><i>i</i></sub></span>) item attributes vary with time
(although one could &quot;trick&quot; the model into allowing that by creating items that are time-specific).</p>
<p>User tastes parameters <span class="formula"><i>β</i><sub><i>u</i>, <i>t</i></sub></span> and <span class="formula"><i>α</i><sub><i>u</i>, <i>t</i></sub></span> are assumed to vary by time period.
This allows for the possibility, for example, that a Netflix viewer might be more inclined
to enjoy a horror movie at night. Or a coffee drinker may prefer espresso drinks more in the
morning than in the evening.</p>
<p><tt class="docutils literal">RWT</tt> requires that <em>time</em> be defined categorically (more on this below). A simple example might be
time = [&quot;Morning&quot;,&quot;Afternoon&quot;,&quot;Evening&quot;]. These categorical labels must be assigned by the researcher prior
to fitting the recommender.</p>
<p>The model is fit using stochastic gradient descent. The equations are identical to those for
<tt class="docutils literal">RWR</tt> with the exception that the <span class="formula"><i>β</i></span> and <span class="formula"><i>α</i></span> parameters are now subscripted with time,
as well.</p>
<div class="section" id="id2">
<h2>Parameters</h2>
<p><tt class="docutils literal">RWT</tt> has the same model parameters as <tt class="docutils literal">RWR</tt>. Parameter arrays <span class="formula"><i>β</i><sub><i>u</i>, <i>t</i></sub></span> and <span class="formula"><i>α</i><sub><i>u</i>, <i>t</i></sub></span>
are identified only using ratings observations for each specific time period.
To account for this, the default value of <strong>n_epochs</strong> has been increased to <tt class="docutils literal">100</tt>.</p>
</div>
<div class="section" id="id3">
<h2>Attributes</h2>
<p>Once an <tt class="docutils literal">RWT</tt> instance is <tt class="docutils literal">fit()</tt>, the resulting parameter values are returned
as attributes of the instance.</p>
<ul class="simple">
<li><dl class="first docutils">
<dt><tt class="docutils literal">intercept_</tt>: (<span class="formula"><i>μ</i></span>)</dt>
<dd>Scalar intercept term</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">bu</tt>: (<span class="formula"><i>b</i><sub><i>u</i></sub></span>)</dt>
<dd>NumPy array with shape (n_users, 1)</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">B</tt>: (<span class="formula"><i>β</i><sub><i>u</i>, <i>t</i></sub></span>)</dt>
<dd>If <span class="formula"><i>X</i><sub><i>i</i></sub></span> is provided to <tt class="docutils literal">fit()</tt> (see below), <tt class="docutils literal">B</tt> is a 3-dimensional NumPy array with shape (n_times, n_users, n_Xs)</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">alpha_</tt>: (<span class="formula"><i>α</i><sub><i>u</i>, <i>t</i></sub></span>)</dt>
<dd>A 3-dimensional NumPy array with shape (n_times, n_users, n_factors)</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">Z</tt>: (<span class="formula"><i>Z</i><sub><i>i</i></sub></span>)</dt>
<dd>NumPy array with shape (u_items, n_factors)</dd>
</dl>
</li>
</ul>
</div>
<div class="section" id="id4">
<h2>Methods</h2>
<ul class="simple">
<li><dl class="first docutils">
<dt><tt class="docutils literal">fit(self,df,Xi=None)</tt></dt>
<dd><ul class="first last">
<li>Fits the recommender system model</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">df</tt> must be a NumPy array</dt>
<dd><ul class="first last">
<li>Each row corresponds to a rating (<span class="formula"><i>r</i><sub><i>uit</i></sub></span>)</li>
<li>Columns <strong>must</strong> be ordered: [user, item, rating, time]</li>
<li><tt class="docutils literal">user</tt> and <tt class="docutils literal">item</tt> may be strings or integers</li>
<li><tt class="docutils literal">time</tt> should be the time label for <span class="formula"><i>r</i><sub><i>uit</i></sub></span>. Can be string or integer but is treated as categorical</li>
</ul>
</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">Xi</tt> (if supplied) must be a NumPy array</dt>
<dd><ul class="first last">
<li>If no observed item attributes are supplied, <tt class="docutils literal">fit()</tt> returns results for SVD with time-varying parameters</li>
<li>First column of <tt class="docutils literal">Xi</tt> <strong>must</strong> be item identifier that corresponds with item labels used in <tt class="docutils literal">df</tt></li>
<li>Shape of array is (n_items, 1 + n_Xs)</li>
</ul>
</dd>
</dl>
</li>
</ul>
</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">accuracy(self,df,Xi=None)</tt></dt>
<dd><ul class="first last">
<li>Returns the mean squared prediction error</li>
<li>Requires the recommender system to be fit first</li>
<li>All provided values must be in the same format as supplied to <tt class="docutils literal">fit()</tt></li>
</ul>
</dd>
</dl>
</li>
<li><dl class="first docutils">
<dt><tt class="docutils literal">predict(self,u_p,i_p,tee)</tt></dt>
<dd><ul class="first last">
<li>Returns predicted ratings</li>
<li><tt class="docutils literal">u_p</tt> is a user value</li>
<li><tt class="docutils literal">i_p</tt> is an item value</li>
<li><tt class="docutils literal">tee</tt> is a time value</li>
<li><tt class="docutils literal">u_p</tt>, <tt class="docutils literal">i_p</tt>, and <tt class="docutils literal">tee</tt> must be provided in the same format as <tt class="docutils literal">fit()</tt></li>
</ul>
</dd>
</dl>
</li>
</ul>
</div>
<div class="section" id="id5">
<h2>Sample Syntax</h2>
<p>If we assume that <tt class="docutils literal">dat</tt> is a NumPy array of ratings data (with time label) and <tt class="docutils literal">att</tt> is a NumPy array
of observed item attributes, we can use the following code:</p>
<pre class="literal-block">
from recommendx import RWT
rwt = RWT(n_factors = 4)
rwt.fit(dat,att)
rwt.accuracy(dat,att)
rwt.predict('userA','item10','AM')
rwt.predict('userA','item10','PM')
</pre>
</div>
</div>
</div>
</body>
</html>
