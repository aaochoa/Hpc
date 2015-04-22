<h1>2D Convolution + Sobel filter</h1>
<h2>Overview</h2>
<p>Here's an implementation of the Sobel filter and a 2D convolution using CUDA C and OpenCV. Also there are three different parallel implementations
using global memmory, constant memory and shared memory</p>
<h3>Computer specs</h3>
<p>&#8594; <span class="caps">CPU</span> </p>
<ul>
	<li>processor       : 1</li>
	<li>vendor_id       : GenuineIntel</li>
	<li>cpu family      : 6</li>
	<li>model           : 58</li>
	<li>model name      : Intel&#174; Core&#8482; i7-3770K <span class="caps">CPU</span> @ 3.50GHz</li>
	<li>stepping        : 9</li>
	<li>cpu MHz         : 1600.000</li>
	<li>cache size      : 8192 KB</li>
	<li>cpu cores       : 4</li>
</ul>
<p>&#8594; <span class="caps">GPU</span> </p>
<ul>
	<li>Tesla K40c: 3.5</li>
	<li>Global memory:   11519mb</li>
	<li>Shared memory:   48kb</li>
	<li>Constant memory: 64kb</li>
	<li>Block registers: 65536</li>
	<li>Warp size:         32</li>
	<li>Threads per block: 1024</li>
	<li>Max block dimensions: [ 1024, 1024, 64 ]</li>
	<li>Max grid dimensions:  [ 2147483647, 65535, 65535 ]</li>
</ul>
<h2>Example</h2>
<p><img src="Images /Source/img1.jpg" alt="Execution" /></p>
<p></p>
<p><img src="Images /Serial/SerialImg1.png" alt="Results" /></p>
