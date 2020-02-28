################
## Exceptions
################

set_propagated_clock [all_clocks]

# CORE MULTICYCLE PATH
# RESET

set_false_path -from [get_ports rst_ni]

# ICACHE PRIVATE

for { set i 0 } { $i < 8 } { incr i } {
  set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q ]  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_rdata_o* ]
  set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q ]  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_rdata_o* ]
  set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_TAG_WAY_[*].TAG_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]     -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_rdata_o* ]
  set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_TAG_WAY_[*].TAG_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]     -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_rdata_o* ]

  set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q ]
  set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q ]
  set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q ]
  set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_DATA_WAY_[*].DATA_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q ]

  set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_TAG_WAY_[*].TAG_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]     -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_rvalid_o* ]
  set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_TAG_WAY_[*].TAG_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]     -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_gnt_o*    ]
  set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_TAG_WAY_[*].TAG_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]     -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_rvalid_o* ]
  set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/_TAG_WAY_[*].TAG_BANK/register_file_1r_1w_i/MemContentxDP_reg*/Q]     -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/PRI_ICACHE[$i].i_pri_icache/fetch_gnt_o*    ]
}

# ICACHE SHARED

set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q]    -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/fetch_rdata_o*]
set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q]    -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/fetch_rdata_o*]

set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/*]    -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q]
set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/*]    -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q]

set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q] -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/fetch_rvalid_o*]
set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q] -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/fetch_gnt_o*]
set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q] -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/fetch_rvalid_o*]
set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/Main_Icache[*].i_main_shared_icache/TAG_RAM_WAY[*].TAG_RAM/scm_tag/register_file_1r_1w_i/MemContentxDP_reg*/Q] -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/gen_priv_icache.icache_top_i/fetch_gnt_o*]

# REGISTER FILE

set_multicycle_path 2 -setup -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/CORE[*].core_region_i/RISCV_CORE/id_stage_i/registers_i/riscv_register_file_i/mem_reg*/Q]
set_multicycle_path 1 -hold  -through [get_pins gen_clusters[0].gen_cluster_sync.i_cluster/i_ooc/i_bound/CORE[*].core_region_i/RISCV_CORE/id_stage_i/registers_i/riscv_register_file_i/mem_reg*/Q]
