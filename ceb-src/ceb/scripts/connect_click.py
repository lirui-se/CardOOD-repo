import clickhouse_driver as chd

def execute_clickhouse(sql, db_name):




sql = "SELECT COUNT(*) FROM catalog_sales AS catalog_sales , date_dim AS d1 , inventory AS inventory , promotion AS promotion , warehouse AS warehouse WHERE  cs_wholesale_cost BETWEEN 27 AND 47 AND  d1.d_year = 1999 AND catalog_sales.cs_item_sk = inventory.inv_item_sk AND catalog_sales.cs_promo_sk = promotion.p_promo_sk AND catalog_sales.cs_sold_date_sk = d1.d_date_sk AND warehouse.w_warehouse_sk = inventory.inv_warehouse_sk;"
