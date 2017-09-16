from setuptools import setup, find_packages

setup(
        name = 'mobula',
        version = '1.0',
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
        install_requires = [
            'numpy',
            'numpy_groupies'
        ]
)
