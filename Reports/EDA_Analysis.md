# Hotel Booking Analysis Report

### 1. **Data Overview**
The dataset used for this analysis contains information about hotel bookings, including factors such as hotel type, lead time, booking status, market segment, and pricing (ADR – Average Daily Rate).

#### **1.1 Data Cleaning and Normalization**
- Data Cleaning is done on the following columns based on our data explorations
  - Children - Used the median to fill the null values
  - is_canceled - Converted to One-shot binary from Yes or No
  - country - Used the mode to fill the null values
  - agent - Explicity cast Unknown in null values
  - company - Explicity cast Unknown in null values 

---

### 2. **Key Insights**

#### **2.1. Revenue Analysis by Hotel Type**
- **Resort Hotels** tend to have a higher average daily rate (ADR) compared to **City Hotels**, indicating that guests are willing to pay more for resort stays, likely due to better amenities, location, and vacation-related services.
- **City Hotels** cater more to business travelers and short stays, leading to relatively lower ADRs.

#### **2.2. Revenue Differences by Market Segment**
- Market segments such as **Corporate** and **Direct** bookings show relatively higher ADRs, indicating that these segments are associated with more premium-priced bookings.
- **Online Travel Agencies (OTAs)**, while providing a significant volume of bookings, tend to have lower ADRs, likely due to the competitive nature of the OTA market and price comparisons available to customers.
- **Groups** bookings generally have lower ADRs, likely because they often receive discounts for bulk bookings.

#### **2.3. Impact of Lead Time on Revenue**
- There is a noticeable relationship between **lead time** (the number of days between the booking date and the arrival date) and ADR.
- **Bookings made far in advance** (longer lead time) tend to have **higher ADRs**, possibly reflecting advance pricing strategies where customers secure higher-priced bookings early. However, the ADR fluctuates depending on the seasonality and demand.
- **Last-minute bookings** (shorter lead time) may lead to slightly lower ADRs, potentially reflecting discounts or promotions offered to fill up remaining room capacity.

#### **2.4. Seasonality Impact on Revenue**
- **Seasonality** plays a crucial role in revenue generation. As observed:
  - **Summer** tends to generate the highest revenue, which is expected as summer is typically a peak season for vacations.
  - **Winter** also shows elevated ADRs, possibly due to the holiday season and increased demand for both resort and city hotels during this time.
  - **Fall** and **Spring** show slightly lower ADRs, indicating these are likely off-peak seasons where demand softens, and hotels may offer competitive pricing to attract guests.

---

### 3. **Booking and Cancellation Patterns**

#### **3.1. Cancellations by Market Segment**
- The **online travel agencies (OTA)** segment experiences the **highest cancellation rates**, likely because OTA customers may have more flexible cancellation policies.
- **Corporate** and **Direct** bookings show much lower cancellation rates, which is consistent with more secure and planned travel (especially business-related).
- **Group bookings** also show moderately higher cancellation rates, possibly because group organizers may change plans based on group size fluctuations.

#### **3.2. Cancellation Trends by Lead Time**
- Bookings with **longer lead times** are more likely to be canceled compared to those made closer to the arrival date. This suggests that guests may cancel early bookings as their travel plans evolve, especially for vacations.
- **Shorter lead time** bookings (last-minute bookings) tend to have fewer cancellations, likely because plans are more confirmed and imminent.

#### **3.3. Cancellation Trends by Season**
- **July** and **August** show elevated cancellation rates, likely reflecting vacation plans being adjusted or changed during peak summer months.
- **December** also shows significant cancellations, which may be related to holiday travel plans that are more flexible or subject to change.

---

### 4. **Conclusion and Recommendations**

Based on the analysis, several key factors influence hotel revenue, including hotel type, market segment, lead time, and seasonality. Here are some recommendations:

- **Revenue Optimization**:
  - For **Resort Hotels**, continuing to leverage premium pricing during peak seasons (Summer, Winter holidays) can maximize revenue.
  - **City Hotels** could explore increasing ADRs during business-centric events, while keeping competitive pricing for short-stay guests.
  - **Dynamic pricing strategies** could be implemented to encourage early bookings with higher rates and offer discounts for last-minute bookings to fill room capacity.

- **Targeted Marketing**:
  - **Direct and Corporate segments** offer more secure bookings with lower cancellation rates, so continued focus on these segments should be prioritized for stable revenue generation.
  - **OTAs**, while contributing to volume, may require better strategies to mitigate high cancellation rates, such as offering non-refundable rates or incentivizing confirmed bookings.

- **Managing Cancellations**:
  - Since **lead time correlates with cancellations**, offering flexible rebooking options or discounts for confirming bookings closer to the stay date could help minimize cancellations.
  - Targeting **re-booking campaigns** for high-cancellation periods (July, August, December) could reduce the impact of lost revenue during peak times.

