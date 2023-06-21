![Symbolator logo](https://zebreus.github.io/symbolator/_static/symbolator_icon.png)

# Symbolator

Symbolator is a component diagramming tool for VHDL and Verilog. It will parse HDL source files, extract components or modules and render them as an image.

This is a fork of [kevinpt/symbolator](https://github.com/kevinpt/symbolator).

```vhdl
library ieee;
use ieee.std_logic_1164.all;

package demo is
  component demo_device is
    generic (
      SIZE : positive;
      RESET_ACTIVE_LEVEL : std_ulogic := '1'
    );
    port (
      --# {{clocks|}}
      Clock : in std_ulogic;
      Reset : in std_ulogic;

      --# {{control|Named section}}
      Enable : in std_ulogic;
      Data_in : in std_ulogic_vector(SIZE-1 downto 0);
      Data_out : out std_ulogic_vector(SIZE-1 downto 0)
    );
  end component;
end package;
```

```console
$ symbolator -i demo_device.vhdl
  Scanning library: .
  Creating symbol for demo_device.vhdl "demo_device"
        -> demo_device-demo_device.svg
```

Produces the following:

![Demo device diagram](https://zebreus.github.io/symbolator/_images/demo_device-demo_device.svg)

Symbolator can render to PNG bitmap images or SVG, PDF, PS, and EPS vector images. SVG is the default.

## Getting symbolator

There are some ways to run symbolator that dont require installing local dependencies:

You can run a prebuilt docker container with symbolator using the following command:

```sh
docker run --rm -it -v $(pwd):/src madmanfred/symbolator
```

Another way to run symbolizer is through the nix package manager with the command:

```sh
nix run github:zebreus/symbolator
```

There also is a AppImage available on GitHub releases.

## Installing symbolator from source

Symbolator is a Python application. You must have Python installed first to use it. Most modern Linux distributions and OS/X have it available by default. There are a number of options available for Windows. If you don't already have a favorite, I recommend getting one of the ["full-stack" Python distros](https://www.scipy.org/install.html) that are geared toward scientific computing such as Anaconda or Python(x,y).

If your OS has a package manager, it may be preferable to install Python setuptools through that tool before attempting to install Symbolator. Otherwise, the installation script will install these packages directly without registering them with the OS package manager.

Symbolator requires Python 3.x, Pycairo, and Pango.

The Pango library is used compute the dimensions of a text layout. There is no standard package to get the Pango Python bindings installed. It is a part of the Gtk+ library which is accessed either through the PyGtk or PyGObject APIs, both of which are supported by Symbolator. You should make sure that one of these libraries is available before installing Symbolator. A [Windows installer](https://www.pygtk.org/downloads.html) is available. For Linux distributions you should install the relevant libraries with your package manager.

To build symbolator from source you need a setuptools with a version >= 61.0.0 and pip >= 23.0.0.

When these requirements are met you can install symbolator from github using:

```sh
pip install --upgrade git+https://github.com/zebreus/symbolator.git
```

If you manually downloaded a source package or created a clone with Git you can install Symbolator with the following command run from the base Symbolator directory:

```sh
pip install .
```

After a successful install the Symbolator command line application will be available. On Linux they should be immediately accessible from your current search path. On Windows you will need to make sure that the `<Python root>\Scripts` directory is in your %PATH% environment variable.

You may need to update to recent versions of setuptools (at least 61) and pip (at least 23) to install symbolator. You can upgrade them to the latest versions using the following command:

```sh
pip install --upgrade pip setuptools
```

On Linux systems you may need to install with root privileges using the *sudo* command.

## Licensing

Symbolator is licensed for free commercial and non-commercial use under the terms of the MIT license. The Symbolator Sphinx extension is derived from the Graphviz extension and is BSD licensed.

## Download

You can access the Symbolator Git repository from [Github](https://github.com/zebreus/symbolator). You can install direct from PyPI with the "pip" command if you have it available.

## Contributing

Thank you for considering contributing to this project!

To contribute, please follow these steps:

1. Fork the repository and create your branch from main.
2. Make the necessary changes and additions.
3. Ensure that you are happy with your changes.
4. Submit a pull request, describing the changes you've made and providing any relevant information.

I will review your pull request as soon as possible.

## Documentation

The full documentation is available online at the [main Symbolator site](https://zebreus.github.io/symbolator/).
