#### *IB->rt_power.readOp.dynamic* 的相关计算
在**Core.cc**中的*InstFetchU::computeEnergy(bool is_tdp)* 函数中：  
```cpp
    IB->rt_power.readOp.dynamic =
        IB->local_result.power.readOp.dynamic * IB->rtp_stats.readAc.access;
    IB->rt_power.readOp.dynamic +=
        IB->local_result.power.writeOp.dynamic * IB->rtp_stats.writeAc.access;
```
该函数在**Core.cc**中的```Core::compute()```和 ```Core::computeEnergy()```函数中以```ifu->computeEnergy(false)```和```ifu->computeEnergy(is_tdp)```的形式被调用。

```Core::compute()```函数在**processor.cc**中的```Processor::compute()```函数中以```cores[i]->compute()```的形式被调用。但是其中```i```遍历```0-numCore```,```numCore```在初始化时就由```numCore = procdynp.numCore == 0 ? 0 : 1;``` 确定为0或1，```numCore```代表的不是核数，而是所有核心为同构。
而在**gpgpu_sim_wrapper.cc**中有```void gpgpu_sim_wrapper::compute() { proc->compute(); }```,在**powerinterface.cc**中的```mcpat_cycle()```函数中调用```wrapper->compute();```

```Core::computeEnergy()```函数在**processor.cc**中的初始化函数中调用：
```cpp
Processor::Processor(ParseXML *XML_interface) {
    cores[i]->computeEnergy();
    cores[i]->computeEnergy(false);
``` 

#### dvfs
在**gpu-sim.cc**中，有
```cpp
  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
```
但是该参数仅影响在**gpgpu_sim_wrapper.cc**中的代码（即下一栏中所示代码），发现： 
除非使用 *HW* 仿真模式，*voltage_ratio*就是*1/1=1*。

#### Core Power
根据**Core.cc**的```displayEnergy```函数在 .o 文件中的输出信息，判断具体的*Core Power*：
包含：  
* IBP
* DCP
* SHRDP
* RFP
* INTP
* FPUP
* INTMULP
* FP_EXP
* SCHEDP
* PIPEP
* IDLE_COREP

不包含：
* MCs Power:
  * MCP
  * DRAMP
* NoCs Power:
  * NOCP
* L2s Power:
  * L2CP
* CONSTP
* STATICP

从**gpgpu_sim_wrapper.cc**中的代码得到，*STATICP*和*IDLE_COREP*与电压是线性关系，应该是属于静态功耗，虽然静态功耗 $P_{static​}∝V⋅e^{kV}$，但此处应该是做了一个近似。
```cpp
  if (g_dvfs_enabled) {
    double voltage_ratio =
        modeled_chip_voltage / p->sys.modeled_chip_voltage_ref;
    sample_cmp_pwr[IDLE_COREP] *=
        voltage_ratio;  // static power scaled by voltage_ratio
    sample_cmp_pwr[STATICP] *=
        voltage_ratio;  // static power scaled by voltage_ratio
    for (unsigned i = 0; i < num_pwr_cmps; i++) {
      if ((i != IDLE_COREP) && (i != STATICP)) {
        sample_cmp_pwr[i] *=
            voltage_ratio *
            voltage_ratio;  // dynamic power scaled by square of voltage_ratio
      }
    }
  }
```

#### power stat相关文件层级关系：
    gpu-sim.cc -> power_interface.cc -> gpgpu_sim_wrapper.cc -> processor.cc -> core.cc

#### Core Power的数量
在**processor.cc**中的```Processor()```构造函数中，有
```cpp
  if (procdynp.homoCore)
    numCore = procdynp.numCore == 0 ? 0 : 1;
  else
    numCore = procdynp.numCore;
```
通过 *cout* 发现 *procdynp.homoCore* 和 *procdynp.numCore* 的值都是1。并且在后续代码中有
```cpp
  for (i = 0; i < numCore; i++) {
    cores.push_back(new Core(XML, i, &interface_ip));
    cores[i]->computeEnergy();
    cores[i]->computeEnergy(false);
    if (procdynp.homoCore) {
      core.area.set_area(core.area.get_area() +
                         cores[i]->area.get_area() * procdynp.numCore);
```
发现他是将所有核心视作一个整体来计算的，最终```proc```中只有一个```core[0]```对象。通过
```cpp
core.area.set_area(core.area.get_area() +
                   cores[i]->area.get_area() * procdynp.numCore
```
可以推测他并没有将```procdynp.numCore```默认为1。该参数可以在**accelwattch_sass_sim.xml**中通过添加```<param name="number_of_cores" value="1"/> <!-- SMs -->```改变。

#### 其他 
在**gpgpusim_wrapper.cc**中，*update_coefficients*函数被调用了，但是通过注释掉对比输出，发现并没有影响功耗计算结果（除STATICP外的power）。
```cpp
void gpgpu_sim_wrapper::update_components_power() {
  update_coefficients();
``` 


经过测试，运行时间基本在*1ms* 内，因此调用*hotspot*计算出的温度基本没有变化是正常的。且每次运行一个*kernel* 都会冷5000个*cycle* 来进行等待（这个参数是在**gpgpusim.config**中的*-gpgpu_kernel_launch_latency* 设置的）

**gpgpu_sim_wrapper.cc**的 *print_power_kernel_stats* 函数将信息打印进了 *accelwattch_power_report.log* 中 

* *CONSTP*不随核心数量变化
* *rt_power.readOp.dynamic*随核心数量变化
* *IDLE_COREP*的值和空闲核心数量线性相关
* *gpu_STATICP* 功能单元被激活并 *ready*，即使不发生大量切换，也要消耗的功率

#### Question
* 将**accelwattch_sass_sim.xml**中的num of cores改为2及以上时，试图创建多个core对象时，报错。
  * 错误信息：```filter_data_arr1.  ERROR: no valid data array organizations found```
  * 代码出错地：**core.cc**中的```  IB = new ArrayST(&interface_ip, "InstBuffer", Core_device, coredynp.opt_local, coredynp.core_ty);```