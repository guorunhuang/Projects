package assignment1.problem1;

public class Deposit {
    private final int amount;
    private final String recipientId;
    private final String recipientName;

    // Constants to avoid magic numbers
    public static final int MIN_TRANSFER = 1000;
    public static final int MAX_TRANSFER = 10000;

    public Deposit(int amount, String recipientId, String recipientName) {
      if (amount < MIN_TRANSFER || amount > MAX_TRANSFER) {
        throw new IllegalArgumentException("Amount must be between 1000 and 10000.");
      }
      this.amount = amount;
      this.recipientId = recipientId;
      this.recipientName = recipientName;
    }

    public int getAmount() { return this.amount; }
    public String getRecipientId() { return this.recipientId; }
    public String getRecipientName() { return this.recipientName; }
}
