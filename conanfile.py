import os

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, CMake, cmake_layout
from conan.tools.build import check_min_cppstd, check_max_cppstd
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy

class ConvolutionalNeuralNetwork(ConanFile):
    name = "cnn"
    version = "1.0"
    
    # optional metadata
    license = ""
    author = "Rajiv Sithiravel rajiv.sithiravel@gmail.com"
    url = ""
    description = ""
    
    # binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False], "optimized": [1, 2, 3]}
    default_options = {"shared": False, "fPIC": True, "optimized": 1}
    
    # sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "cnn/*"
        
    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
            
    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
    
    # operating system validation
    def validate(self):
        print(self.settings.os)
        if self.settings.os != "Windows" & self.settings.os != "Linux" & self.settings.os != "iOS" & self.settings.os != "watchOS" & self.settings.os != "tvOS" & self.settings.os != "visionOS" & self.settings.os != "Macos" & self.settings.os != "Android" & self.settings.os != "FreeBSD" & self.settings.os != "SunOS" & self.settings.os != "AIX" & self.settings.os != "Arduino" & self.settings.os != "Emscripten" & self.settings.os != "Neutrino" & self.settings.os != "baremetal" & self.settings.os != "VxWorks":
                raise ConanInvalidConfiguration("No support for the operating system")
    
    # compiler validations     
    def validate(self):
        print(self.settings.compiler)
        if self.settings.compiler == "sun-cc":
            self.settings.compiler.libcxx = "libstdc++"
        elif self.settings.compiler == "gcc":
            check_min_cppstd(self, "gnu11")   
            check_max_cppstd(self, "gnu23")            
            self.settings.compiler.libcxx = "libstdc++11"
            self.settings.compiler.cppstd = "gnu23"
        elif self.settings.compiler == "msvc":
            check_min_cppstd(self, "14")    
            check_max_cppstd(self, "23")                
            self.settings.compiler.cppstd = "14"
        elif self.settings.compiler == "clang":
            check_min_cppstd(self, "gnu11")    
            check_max_cppstd(self, "gnu23")  
            self.settings.compiler.libcxx = "libstdc++11"
            self.settings.compiler.cppstd = "gnu23"
        elif self.settings.compiler == "apple-clang":
            check_min_cppstd(self, "gnu11")    
            check_max_cppstd(self, "gnu23")  
            self.settings.compiler.libcxx = "libstdc++"
            self.settings.compiler.cppstd = "gnu23"
        elif self.settings.compiler == "intel-cc":
            check_min_cppstd(self, "gnu11")    
            check_max_cppstd(self, "gnu23")     
            self.settings.compiler.mode = "dpcpp"
            self.settings.compiler.libcxx = "ibstdc++11"
            self.settings.compiler.cppstd = "gnu23"
        elif self.settings.compiler == "qcc":   
            check_min_cppstd(self, "gnu11")    
            check_max_cppstd(self, "gnu17")    
            self.settings.compiler.libcxx = "cxx"
            self.settings.compiler.cppstd = "gnu17"
        elif self.settings.compiler == "mcst-lcc":
            check_min_cppstd(self, "gnu11")    
            check_max_cppstd(self, "gnu23") 
            self.settings.compiler.libcxx = "libstdc++11"
            self.settings.compiler.cppstd = "gnu23"
    
    # Needed libraries
    def requirements(self):
        self.requires("boost/1.85.0")
        self.requires("eigen/3.4.0")
        self.requires("opencv/4.9.0")
        
    def build_requirements(self):
        self.tool_requires("cmake/3.30.1")
    
    def layout(self):
        cmake_layout(self)
          
    def generate(self):
        if self.settings.os == "Windows" or self.settings.os == "Linux" :
            tc = CMakeToolchain(self)
            tc.generate()
            deps = CMakeDeps(self)
            deps.generate()
        else:
            tc = AutotoolsToolchain(self)
            tc.generate()
            deps = PkgConfigDeps(self)
            deps.generate()
            
    def build(self):
        if self.settings.os == "Windows" or self.settings.os == "Linux" :
            cmake = CMake(self)
            cmake.configure()
            cmake.build()
        else:
            autotools = Autotools(self)
            autotools.autoreconf()
            autotools.configure()
            autotools.make()

    def package(self):
        cmake = CMake(self)
        cmake.install()
    