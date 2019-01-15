from distutils.core import setup, Extension

module1 = Extension('lda_sta',
                    define_macros = [
                            # ('DEBUG', '1'),
                            # ('DEBUG_TIME', '1'),
                        ],
                    extra_compile_args = ["-Wno-sign-compare"],
                    sources = ['lda_sta.cpp'])

setup (name = 'lda_sta',
       version = '1.0',
       description = 'LDA topic reconstruction based on STA reduction',
       ext_modules = [module1])