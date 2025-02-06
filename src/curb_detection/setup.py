from setuptools import setup

package_name = 'curb_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[],
    install_requires=['setuptools','rclpy','sensor_msgs','numpy','pclpy'],
    zip_safe=True,
    maintainer='Tianchen Wang',  # 修改为你的名字
    maintainer_email='tcgiantmonkey@gmail.com',  # 修改为你的邮箱
    description='ROS2 package for curb detection',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'curb_detector = curb_detection.curb_detector:main', # 注意：这里需要使用相对路径
        ],
    },
)