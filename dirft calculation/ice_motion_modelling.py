# 我们进一步深入讨论海冰运动参数（即速度、方向和角速度）的变化建模。假设我们以以delta_t为时间窗口将整个记录时间分段，共分为n段。设每一段中的平均速度为v_i、方向为\theta_i、角速度为\omega_i。我希望用某个数学模型来描述速度、方向和角速度随时间的变化。即v_i = f(i), \theta_i = g(i), \omega_i = h(i)。请问如何选择数学模型？如线性、二次曲线等。如何使用实际海洋浮标观测数据来帮助数学模型的选择和验证？请给出思路，先不要写程序。



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score
from statsmodels.tools.eval_measures import aicc
from drift_utils import load_and_preprocess_data, calculate_common_timeline, convert_to_polar_coordinates, calculate_parameters

base_interval = '1h'

class IceMotionModeler:
    def __init__(self, folder_path, base_interval='1h'):
        self.dfs = load_and_preprocess_data(folder_path)
        self.time_grid = calculate_common_timeline(self.dfs, base_interval)
        self.polar_dfs = convert_to_polar_coordinates(self.dfs, self.time_grid)
        self.param_df = calculate_parameters(self.polar_dfs, self.time_grid, base_interval)
        
    def _exploratory_analysis(self, series):
        """执行探索性数据分析"""
        # 平稳性检验
        adf_result = adfuller(series)
        # 趋势强度计算（线性趋势的斜率绝对值）
        if len(series) > 1:
            time_index = np.arange(len(series))
            coeffs = np.polyfit(time_index, series, 1)
            trend_strength = abs(coeffs)  # 取一次项系数
        else:
            trend_strength = 0.0
        trend_strength = trend_strength[0]  # 转换为标量
        return {
            'stationary': adf_result[1] < 0.05,
            'trend_strength': float(trend_strength),  # 转换为Python float类型
            'mean': float(np.mean(series)),
            'std': float(np.std(series))
        }

    def _fit_models(self, X, y):
        """修正后的模型拟合方法"""
        models = {}
        
        # 线性模型（1次多项式）
        linear_coeffs = np.polyfit(X, y, 1)
        models['Linear'] = {
            'func': np.poly1d(linear_coeffs),
            'n_params': len(linear_coeffs)  # 参数数=系数数量（斜率和截距）
        }

        # 二次模型（2次多项式）
        quad_coeffs = np.polyfit(X, y, 2)
        models['Quadratic'] = {
            'func': np.poly1d(quad_coeffs),
            'n_params': len(quad_coeffs)
        }

        # 三角函数模型（带参数数量修正）
        try:
            from scipy.optimize import curve_fit
            def _trig_model(x, A, freq, phi):
                return A * np.sin(2 * np.pi * freq * x + phi)
            
            # 参数初始猜测
            p0 = [np.std(y), 1/len(X), 0]
            params, _ = curve_fit(_trig_model, X, y, p0=p0)
            
            models['Trigonometric'] = {
                'func': lambda x: _trig_model(x, *params),
                'n_params': len(params)
            }
        except Exception as e:
            print(f"三角函数拟合失败: {str(e)}")

        results = {}
        for name, model_info in models.items():
            pred = model_info['func'](X)
            residuals = y - pred
            
            # 关键修正点：确保计算单个AICc值
            aicc_val = aicc(
                np.sum(residuals**2),  # SSR (Sum of Squared Residuals)
                len(y),                # 观测数量
                model_info['n_params'] # 参数数量
            )
            
            results[name] = {
                'r2': r2_score(y, pred),
                'aicc': aicc_val,  # 现在为标量值
                'residuals': residuals
            }
        return results

    def analyze_parameter(self, param_name, plot=True):
        """参数分析与建模（完整修复）"""
        series = self.param_df[param_name].dropna()
        if len(series) < 2:
            print(f"参数 {param_name} 数据不足，至少需要2个观测值")
            return None
            
        X = np.arange(len(series))
        y = series.values
        
        # 探索性分析
        eda = self._exploratory_analysis(y)
        print(f"\n{param_name} 分析结果:")
        print(f"平稳性: {'是' if eda['stationary'] else '否'}")
        print(f"趋势强度: {eda['trend_strength']:.2e}")
        
        # 模型拟合
        model_results = self._fit_models(X, y)
        
        # 选择最佳模型
        best_model = min(model_results.items(), key=lambda x: x[1]['aicc'])
        print(f"\n最佳模型: {best_model}")
        print(f"R²: {best_model[1]['r2']:.3f}")
        print(f"AICc: {best_model[1]['aicc']:.1f}")
        
        # 可视化修复
        if plot and len(y) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(y, 'o-', label='实际值', markersize=5)
            
            for name, res in model_results.items():
                predicted = y - res['residuals']
                plt.plot(predicted, '--', linewidth=1.5, label=f'{name}拟合')
                
            plt.title(f'{param_name} 建模结果\n最佳模型: {best_model}')
            plt.xlabel('时间步')
            plt.ylabel(param_name)
            plt.legend()
            plt.grid(True)
            plt.show()
            
        return model_results

    def full_analysis(self):
        """全参数分析流程"""
        results = {}
        for param in ['speed_mps', 'direction_deg', 'angular_speed_degps']:
            results[param] = self.analyze_parameter(param)
        return results

    def cross_validate(self, param_name, model_type='Linear', test_size=0.2):
        """时间序列交叉验证"""
        series = self.param_df[param_name].dropna().values
        split_idx = int(len(series)*(1-test_size))
        
        # 训练集拟合
        train_X = np.arange(split_idx)
        train_y = series[:split_idx]
        model = np.poly1d(np.polyfit(train_X, train_y, 1 if model_type=='Linear' else 2))
        
        # 测试集预测
        test_X = np.arange(split_idx, len(series))
        pred_y = model(test_X)
        test_y = series[split_idx:]
        
        # 评估指标
        mse = np.mean((pred_y - test_y)**2)
        r2 = r2_score(test_y, pred_y)
        
        # 可视化
        plt.figure(figsize=(12,6))
        plt.plot(np.concatenate([train_X, test_X]), 
                np.concatenate([train_y, test_y]), label='实际值')
        plt.plot(test_X, pred_y, 'r--', label='预测值')
        plt.title(f'{param_name} 交叉验证 (MSE={mse:.2e}, R²={r2:.2f})')
        plt.legend()
        plt.show()
        
        return {'mse': mse, 'r2': r2}

# 使用示例
if __name__ == "__main__":
    # 初始化分析器（输入数据文件夹路径）
    folder_path = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\0830_冰基长基线浮标GNSS_O文件转换\解算数据"
    modeler = IceMotionModeler(folder_path,base_interval)
    
    # 执行全参数分析
    full_results = modeler.full_analysis()
    
    # 交叉验证示例
    speed_cv = modeler.cross_validate('speed_mps', model_type='Quadratic')
    angular_cv = modeler.cross_validate('angular_speed_degps')