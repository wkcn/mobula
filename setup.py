from setuptools import setup, find_packages

setup(
        name = 'mobula',
        version = '1.0.1',
        description = 'A Lightweight & Flexible Deep Learning (Neural Network) Framework in Python',
        author = 'wkcn',
        author_email = 'wkcn@live.cn',
        url = 'https://github.com/wkcn/mobula',
        packages = find_packages(),
        package_data = {
            '' : ['*.md'],
            'docs' : ['docs/*.md'],
            'examples' : ['examples/*.py']
        },
        keywords = 'Deep Learning Framework in Python',
        license = 'MIT',
        classifiers = [
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Mathematics',
            'License :: OSI Approved :: MIT License'
        ],
        install_requires = [
            'numpy',
            'numpy_groupies'
        ]
)
