package assignment1.problem1;

public class MilesBalance {
      private int totalMiles;
      private int milesEarnedThisYear;
      private int milesExpiringThisYear;

      public MilesBalance(int total, int earned, int expiring){
        this.totalMiles = total;
        this.milesEarnedThisYear = earned;
        this.milesExpiringThisYear = expiring;
      }

      public void addMiles(int amount) {
        this.totalMiles += amount;
        this.milesEarnedThisYear += amount;
        this.milesExpiringThisYear += amount;
      }

      public void deductMiles(int amount){
        this.totalMiles -= amount;
      }

      public int getTotalMiles() { return this.totalMiles; }
      public int getMilesEarnedThisYear() { return this.milesEarnedThisYear; }
      public int getMilesExpiringThisYear() { return this.milesExpiringThisYear; }
}